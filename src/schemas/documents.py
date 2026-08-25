from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=50_000)

    @field_validator("title", "content")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain non-whitespace characters")
        return value


class DocumentResponse(BaseModel):
    id: UUID
    client_id: UUID
    title: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
