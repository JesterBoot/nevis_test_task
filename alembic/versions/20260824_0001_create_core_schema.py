from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "b10000000001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(sa.DDL("CREATE EXTENSION IF NOT EXISTS vector"))

    op.create_table(
        "clients",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("first_name", sa.String(length=255), nullable=False),
        sa.Column("last_name", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("normalized_email", sa.String(length=320), nullable=False),
        sa.Column("email_domain", sa.String(length=255), nullable=False),
        sa.Column("email_domain_base", sa.String(length=255), nullable=False),
        sa.Column(
            "country_of_residence",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "normalized_email",
            name="uq_clients_normalized_email",
        ),
    )
    op.create_index(
        "ix_clients_email_domain",
        "clients",
        ["email_domain"],
    )
    op.create_index(
        "ix_clients_email_domain_base",
        "clients",
        ["email_domain_base"],
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("client_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["client_id"],
            ["clients.id"],
            name="fk_documents_client_id_clients",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_documents_client_id",
        "documents",
        ["client_id"],
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("document_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.CheckConstraint(
            "position >= 0",
            name="ck_document_chunks_position_non_negative",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "position",
            name="uq_document_chunks_document_position",
        ),
    )
    op.create_index(
        "ix_document_chunks_document_id",
        "document_chunks",
        ["document_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_chunks_document_id",
        table_name="document_chunks",
    )
    op.drop_table("document_chunks")

    op.drop_index("ix_documents_client_id", table_name="documents")
    op.drop_table("documents")

    op.drop_index("ix_clients_email_domain_base", table_name="clients")
    op.drop_index("ix_clients_email_domain", table_name="clients")
    op.drop_table("clients")

    op.execute(sa.DDL("DROP EXTENSION IF EXISTS vector"))
