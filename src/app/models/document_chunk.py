import uuid
from typing import TYPE_CHECKING, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlmodel import Column, Field, Relationship, SQLModel

if TYPE_CHECKING:
    from app.models import Document


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"
    __table_args__ = (
        CheckConstraint(
            "position >= 0",
            name="ck_document_chunks_position_non_negative",
        ),
        UniqueConstraint(
            "document_id",
            "position",
            name="uq_document_chunks_document_position",
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    document_id: uuid.UUID = Field(
        sa_column=Column(
            Uuid(as_uuid=True),
            ForeignKey(
                "documents.id",
                name="fk_document_chunks_document_id_documents",
                ondelete="CASCADE",
            ),
            nullable=False,
            index=True,
        )
    )
    position: int = Field(sa_column=Column(Integer, nullable=False))
    content: str = Field(sa_column=Column(Text, nullable=False))
    embedding: list[float] = Field(
        sa_column=Column(Vector(384), nullable=False)
    )

    document: Optional["Document"] = Relationship(back_populates="chunks")
