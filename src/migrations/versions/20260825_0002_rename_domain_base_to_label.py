from collections.abc import Sequence

from alembic import op

revision: str = "b20000000002"
down_revision: str | None = "b10000000001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_clients_email_domain_base", table_name="clients")
    op.alter_column(
        "clients",
        "email_domain_base",
        new_column_name="email_domain_label",
    )
    op.create_index(
        "ix_clients_email_domain_label",
        "clients",
        ["email_domain_label"],
    )


def downgrade() -> None:
    op.drop_index("ix_clients_email_domain_label", table_name="clients")
    op.alter_column(
        "clients",
        "email_domain_label",
        new_column_name="email_domain_base",
    )
    op.create_index(
        "ix_clients_email_domain_base",
        "clients",
        ["email_domain_base"],
    )
