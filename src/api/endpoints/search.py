from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from core.config import Settings, get_settings
from db.session import AsyncSession, get_session
from schemas.search import (
    ClientSearchResult,
    DocumentSearchResult,
    SearchResult,
)
from search.dependencies import get_embedding_provider
from search.embeddings import EmbeddingProvider
from search.types import ClientSearchMatch
from services.search import search_clients_and_documents

router = APIRouter(prefix="/search", tags=["Search"])


@router.get(
    "",
    response_model=list[SearchResult],
    summary="Search clients and documents",
)
async def search_endpoint(
    query: Annotated[str, Query(alias="q", min_length=1)],
    limit: Annotated[int | None, Query(ge=1)] = None,
    session: Annotated[AsyncSession, Depends(get_session)] = None,
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,
) -> list[SearchResult]:
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Search query must contain non-whitespace characters.",
        )

    effective_limit = (
        settings.search_limit_default if limit is None else limit
    )
    if effective_limit > settings.search_limit_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Search limit must not exceed "
                f"{settings.search_limit_max}."
            ),
        )

    matches = await search_clients_and_documents(
        session=session,
        query=query,
        limit=effective_limit,
        provider=provider,
        settings=settings,
    )

    results: list[SearchResult] = []
    for match in matches:
        if isinstance(match, ClientSearchMatch):
            results.append(ClientSearchResult.model_validate(match))
        else:
            results.append(DocumentSearchResult.model_validate(match))
    return results
