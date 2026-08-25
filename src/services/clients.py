from pydantic import EmailStr
from sqlalchemy.exc import IntegrityError

from db.session import AsyncSession
from models import Client
from schemas.clients import ClientCreate


class DuplicateClientEmailError(Exception):
    """Raised when the database rejects a duplicate normalized email."""


def normalize_email(email: EmailStr | str) -> tuple[str, str, str]:
    normalized_email = str(email).strip().lower()
    _, domain = normalized_email.rsplit("@", 1)
    domain_label = domain.rsplit(".", 1)[0] if "." in domain else domain
    return normalized_email, domain, domain_label


async def create_client(
    session: AsyncSession,
    payload: ClientCreate,
) -> Client:
    normalized_email, email_domain, email_domain_label = normalize_email(
        payload.email
    )
    client = Client(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=normalized_email,
        normalized_email=normalized_email,
        email_domain=email_domain,
        email_domain_label=email_domain_label,
        country_of_residence=payload.country_of_residence,
    )
    session.add(client)

    try:
        await session.commit()
    except IntegrityError as exc:
        raise DuplicateClientEmailError from exc

    await session.refresh(client)
    return client


__all__ = (
    "DuplicateClientEmailError",
    "create_client",
    "normalize_email",
)
