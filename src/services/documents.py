from uuid import UUID

from core.config import Settings, get_settings
from db.session import AsyncSession
from models import Client, Document, DocumentChunk
from schemas.documents import DocumentCreate
from search.chunking import chunk_text
from search.embeddings import EmbeddingProvider
from search.types import EmbeddedChunk


class ClientNotFoundError(Exception):
    """Client does not exist."""


class DocumentValidationError(Exception):
    """Raised when configured document limits are exceeded."""


async def create_document(
    session: AsyncSession,
    client_id: UUID,
    payload: DocumentCreate,
    provider: EmbeddingProvider,
    *,
    settings: Settings | None = None,
) -> Document:
    resolved_settings = settings or get_settings()
    client = await session.get(Client, client_id)
    if client is None:
        raise ClientNotFoundError

    if session.in_transaction():
        await session.rollback()

    try:
        chunks = chunk_text(
            payload.content,
            max_document_chars=resolved_settings.max_document_chars,
            max_chunks=resolved_settings.max_chunks,
            chunk_size=resolved_settings.chunk_size,
            chunk_overlap=resolved_settings.chunk_overlap,
        )
    except ValueError as exc:
        raise DocumentValidationError(str(exc)) from exc

    embedding_inputs = [
        _document_chunk_embedding_text(payload.title, chunk.content)
        for chunk in chunks
    ]
    embeddings = provider.embed(embedding_inputs)
    if len(embeddings) != len(chunks):
        raise ValueError("embedding provider returned an invalid batch length")

    embedded_chunks = [
        EmbeddedChunk(
            position=chunk.position,
            content=chunk.content,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    document = Document(
        client_id=client_id,
        title=payload.title,
        content=payload.content,
    )
    document.chunks = [
        DocumentChunk(
            position=chunk.position,
            content=chunk.content,
            embedding=chunk.embedding,
        )
        for chunk in embedded_chunks
    ]

    async with session.transaction():
        session.add(document)
        await session.flush()

    await session.refresh(document)
    return document


def _document_chunk_embedding_text(title: str, chunk_content: str) -> str:
    return f"{title}\n\n{chunk_content}"
