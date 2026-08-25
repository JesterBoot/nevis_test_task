import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Column, DateTime, Field, Relationship, SQLModel, func

if TYPE_CHECKING:
    from models import Document


class Client(SQLModel, table=True):
    __tablename__ = "clients"
    __table_args__ = (
        UniqueConstraint(
            "normalized_email",
            name="uq_clients_normalized_email",
        ),
        Index("ix_clients_email_domain", "email_domain"),
        Index("ix_clients_email_domain_label", "email_domain_label"),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    first_name: str = Field(max_length=255)
    last_name: str = Field(max_length=255)
    email: str = Field(max_length=320)
    normalized_email: str = Field(max_length=320)
    email_domain: str = Field(max_length=255)
    email_domain_label: str = Field(max_length=255)
    country_of_residence: str | None = Field(default=None, max_length=255)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )

    documents: list["Document"] = Relationship(
        back_populates="client",
        sa_relationship_kwargs={
            "cascade": "all, delete-orphan",
            "passive_deletes": True,
        },
    )
