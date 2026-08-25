from collections.abc import Iterable
from uuid import UUID

from sqlalchemy import case, cast, func, literal, or_
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlmodel import select

from core.config import Settings, get_settings
from db.session import AsyncSession
from models import Client, Document, DocumentChunk
from search.embeddings import EmbeddingProvider
from search.normalization import (
    candidate_limit,
    normalize_search_query,
)
from search.query import NormalizedSearchQuery
from search.types import (
    ClientSearchMatch,
    DocumentRankingCandidate,
    DocumentSearchMatch,
)

SEARCH_CONFIG = cast(literal("simple"), REGCONFIG)


async def search_clients_and_documents(
    session: AsyncSession,
    query: str,
    limit: int,
    provider: EmbeddingProvider,
    *,
    settings: Settings | None = None,
) -> list[ClientSearchMatch | DocumentSearchMatch]:
    resolved_settings = settings or get_settings()
    normalized_query = normalize_search_query(query)

    clients = await _search_clients(session, normalized_query)
    documents = await _search_documents(
        session,
        normalized_query,
        limit=limit,
        provider=provider,
        settings=resolved_settings,
    )

    combined: list[ClientSearchMatch | DocumentSearchMatch] = [
        *clients,
        *documents,
    ]
    return combined[:limit]


async def _search_clients(
    session: AsyncSession,
    query: NormalizedSearchQuery,
) -> list[ClientSearchMatch]:
    exact_conditions = []
    if query.domain is not None:
        exact_conditions.append(Client.email_domain == query.domain)
    if query.compact:
        exact_conditions.append(Client.email_domain_label == query.compact)

    if exact_conditions:
        exact_rank = case(
            (Client.email_domain == query.domain, 0)
            if query.domain is not None
            else (literal(False), 1),
            else_=1,
        )
        exact_statement = (
            select(Client)
            .where(or_(*exact_conditions))
            .order_by(exact_rank, Client.id)
        )
        exact_clients = (await session.exec(exact_statement)).all()
        if exact_clients:
            return [_to_client_match(client) for client in exact_clients]

    vector = _client_search_vector()
    ts_query = func.plainto_tsquery(SEARCH_CONFIG, query.normalized)
    rank = func.ts_rank_cd(vector, ts_query)
    fts_statement = (
        select(Client)
        .where(vector.op("@@")(ts_query))
        .order_by(rank.desc(), Client.id)
    )
    clients = (await session.exec(fts_statement)).all()
    return [_to_client_match(client) for client in clients]


async def _search_documents(
    session: AsyncSession,
    query: NormalizedSearchQuery,
    *,
    limit: int,
    provider: EmbeddingProvider,
    settings: Settings,
) -> list[DocumentSearchMatch]:
    query_embedding = _embed_query(provider, query.raw)
    bounded_limit = candidate_limit(limit)

    vector_distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    vector_statement = (
        select(DocumentChunk.id)
        .order_by(vector_distance, DocumentChunk.id)
        .limit(bounded_limit)
    )
    vector_chunk_ids = {
        chunk_id for chunk_id in (await session.exec(vector_statement)).all()
    }

    lexical_document_ids = await _lexical_document_ids(
        session,
        query,
        bounded_limit,
    )
    lexical_chunk_ids: set[UUID] = set()
    if lexical_document_ids:
        lexical_chunks_statement = select(DocumentChunk.id).where(
            DocumentChunk.document_id.in_(lexical_document_ids)
        )
        lexical_chunk_ids = {
            chunk_id
            for chunk_id in (await session.exec(lexical_chunks_statement)).all()
        }

    candidate_chunk_ids = vector_chunk_ids | lexical_chunk_ids
    if not candidate_chunk_ids:
        return []

    final_distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    raw_cosine = (literal(1.0) - final_distance).label("raw_cosine")
    scored_statement = (
        select(DocumentChunk, raw_cosine, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.id.in_(candidate_chunk_ids))
        .where(raw_cosine >= settings.semantic_similarity_threshold)
        .order_by(
            Document.id,
            raw_cosine.desc(),
            DocumentChunk.position,
            DocumentChunk.id,
        )
    )
    rows = (await session.exec(scored_statement)).all()

    matches = rank_document_candidates(
        (
            (
                document.id,
                document.client_id,
                document.title,
                chunk.content,
                float(similarity),
                document.id in lexical_document_ids,
                chunk.position,
                chunk.id,
            )
            for chunk, similarity, document in rows
        ),
        threshold=settings.semantic_similarity_threshold,
        fts_boost=settings.fts_boost,
        snippet_length=settings.search_snippet_length,
    )
    return matches


def rank_document_candidates(
    rows: Iterable[tuple[UUID, UUID, str, str, float, bool, int, UUID]],
    *,
    threshold: float,
    fts_boost: float,
    snippet_length: int,
) -> list[DocumentSearchMatch]:
    candidates: dict[UUID, DocumentRankingCandidate] = {}
    for (
        document_id,
        client_id,
        title,
        content,
        similarity,
        lexical_match,
        chunk_position,
        chunk_id,
    ) in rows:
        if similarity < threshold:
            continue

        existing = candidates.get(document_id)
        if existing is None:
            candidates[document_id] = DocumentRankingCandidate(
                id=document_id,
                client_id=client_id,
                title=title,
                snippet=_snippet(content, snippet_length),
                best_raw_cosine=similarity,
                lexical_match=lexical_match,
                best_chunk_position=chunk_position,
                best_chunk_id=chunk_id,
            )
            continue

        existing.lexical_match = existing.lexical_match or lexical_match
        is_better_chunk = similarity > existing.best_raw_cosine or (
            similarity == existing.best_raw_cosine
            and (chunk_position, str(chunk_id))
            < (existing.best_chunk_position, str(existing.best_chunk_id))
        )
        if is_better_chunk:
            existing.best_raw_cosine = similarity
            existing.snippet = _snippet(content, snippet_length)
            existing.best_chunk_position = chunk_position
            existing.best_chunk_id = chunk_id

    matches = [
        DocumentSearchMatch(
            id=candidate.id,
            client_id=candidate.client_id,
            title=candidate.title,
            snippet=candidate.snippet,
            ranking_score=(
                candidate.best_raw_cosine + fts_boost
                if candidate.lexical_match
                else candidate.best_raw_cosine
            ),
        )
        for candidate in candidates.values()
    ]
    return sorted(
        matches,
        key=lambda match: (-match.ranking_score, str(match.id)),
    )


async def _lexical_document_ids(
    session: AsyncSession,
    query: NormalizedSearchQuery,
    limit: int,
) -> set[UUID]:
    vector = _document_search_vector()
    ts_query = func.plainto_tsquery(SEARCH_CONFIG, query.normalized)
    rank = func.ts_rank_cd(vector, ts_query)
    statement = (
        select(Document.id)
        .where(vector.op("@@")(ts_query))
        .order_by(rank.desc(), Document.id)
        .limit(limit)
    )
    return {
        document_id for document_id in (await session.exec(statement)).all()
    }


def _client_search_vector():
    return func.to_tsvector(
        SEARCH_CONFIG,
        func.concat_ws(
            " ",
            Client.first_name,
            Client.last_name,
            Client.email,
            Client.email_domain,
            Client.email_domain_label,
        ),
    )


def _document_search_vector():
    return func.to_tsvector(
        SEARCH_CONFIG,
        func.concat_ws(" ", Document.title, Document.content),
    )


def _embed_query(provider: EmbeddingProvider, query: str) -> list[float]:
    embeddings = provider.embed([query])
    if len(embeddings) != 1:
        raise ValueError("embedding provider returned an invalid query batch")
    return embeddings[0]


def _to_client_match(client: Client) -> ClientSearchMatch:
    return ClientSearchMatch(
        id=client.id,
        first_name=client.first_name,
        last_name=client.last_name,
        email=client.email,
    )


def _snippet(content: str, max_length: int) -> str:
    if len(content) <= max_length:
        return content
    return f"{content[:max_length]}..."
