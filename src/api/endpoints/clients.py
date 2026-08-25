from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from db.session import AsyncSession, get_session
from schemas.clients import ClientCreate, ClientResponse
from services.clients import DuplicateClientEmailError, create_client

router = APIRouter()


@router.post(
    "",
    response_model=ClientResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
    summary="Create a client",
    tags=["Clients"],
)
async def create_client_endpoint(
    payload: ClientCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ClientResponse:
    try:
        client = await create_client(session, payload)
    except DuplicateClientEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A client with this email already exists.",
        ) from exc

    return ClientResponse.model_validate(client)
