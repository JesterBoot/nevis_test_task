import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Text, Uuid
from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

if TYPE_CHECKING:
    from models import Client, DocumentChunk


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    client_id: uuid.UUID = Field(
        sa_column=Column(
            Uuid(as_uuid=True),
            ForeignKey(
                "clients.id",
                name="fk_documents_client_id_clients",
                ondelete="CASCADE",
            ),
            nullable=False,
            index=True,
        )
    )
    title: str = Field(max_length=255)
    content: str = Field(sa_column=Column(Text, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )

    client: Optional["Client"] = Relationship(back_populates="documents")
    chunks: list["DocumentChunk"] = Relationship(
        back_populates="document",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "passive_deletes": True,
        },
    )
