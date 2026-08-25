from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status

from db.session import AsyncSession, get_session
from schemas.documents import DocumentCreate, DocumentResponse
from search.dependencies import get_embedding_provider
from search.embeddings import EmbeddingProvider
from services.documents import (
    ClientNotFoundError,
    DocumentValidationError,
    create_document,
)

router = APIRouter()


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a document",
    tags=["Documents"],
)
async def create_document_endpoint(
    client_id: Annotated[
        UUID,
        Path(alias="id", description="The parent client identifier"),
    ],
    payload: DocumentCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> DocumentResponse:
    try:
        document = await create_document(
            session,
            client_id,
            payload,
            provider,
        )
    except ClientNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Client not found.",
        ) from exc
    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return DocumentResponse.model_validate(document)
