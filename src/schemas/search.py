from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ClientSearchResult(BaseModel):
    type: Literal["client"] = "client"
    id: UUID
    first_name: str
    last_name: str
    email: str

    model_config = ConfigDict(from_attributes=True)


class DocumentSearchResult(BaseModel):
    type: Literal["document"] = "document"
    id: UUID
    client_id: UUID
    title: str
    snippet: str

    model_config = ConfigDict(from_attributes=True)


SearchResult = Annotated[
    ClientSearchResult | DocumentSearchResult,
    Field(discriminator="type"),
]
