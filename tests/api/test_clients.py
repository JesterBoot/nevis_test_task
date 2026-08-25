from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import delete

from core.config import Settings
from db.session import AsyncSession, get_session
from main import create_app
from models import Client, Document, DocumentChunk
from services.clients import normalize_email

DATABASE_URL = Settings().database_url
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not DATABASE_URL.startswith("postgresql"),
        reason="Client API tests require PostgreSQL",
    ),
]


@pytest_asyncio.fixture
async def api_app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async with session_factory() as session:
        await session.exec(delete(DocumentChunk))
        await session.exec(delete(Document))
        await session.exec(delete(Client))
        await session.commit()

    async def override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    application = create_app()
    application.dependency_overrides[get_session] = override_get_session
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
def api_client(api_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(api_app) as client:
        yield client


def test_create_client_normalizes_email_and_preserves_public_contract(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/clients",
        json={
            "first_name": "Anton",
            "last_name": "Batiaev",
            "email": " Anton.Batiaev@NevisWealth.com ",
            "countryOfResidence": "ME",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["first_name"] == "Anton"
    assert body["last_name"] == "Batiaev"
    assert body["email"] == "anton.batiaev@neviswealth.com"
    assert body["countryOfResidence"] == "ME"
    assert "normalized_email" not in body
    assert "email_domain" not in body
    assert "email_domain_label" not in body


def test_normalize_email_derives_complete_domain_and_lightweight_label() -> None:
    assert normalize_email(" Anton.Batiaev@NevisWealth.com ") == (
        "anton.batiaev@neviswealth.com",
        "neviswealth.com",
        "neviswealth",
    )


def test_duplicate_normalized_email_returns_conflict(
    api_client: TestClient,
) -> None:
    first_response = api_client.post(
        "/clients",
        json={
            "first_name": "Anton",
            "last_name": "Batiaev",
            "email": "anton@example.com",
        },
    )
    second_response = api_client.post(
        "/clients",
        json={
            "first_name": "Another",
            "last_name": "Person",
            "email": " ANTON@EXAMPLE.COM ",
        },
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert second_response.json()["detail"] == (
        "A client with this email already exists."
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "first_name": "Anton",
            "last_name": "Batiaev",
            "email": "not-an-email",
        },
        {
            "first_name": "Anton",
            "email": "anton@example.com",
        },
        {
            "first_name": "",
            "last_name": "Batiaev",
            "email": "anton@example.com",
        },
    ],
)
def test_invalid_client_payload_returns_unprocessable_entity(
    api_client: TestClient,
    payload: dict[str, str],
) -> None:
    response = api_client.post("/clients", json=payload)

    assert response.status_code == 422


def test_openapi_exposes_client_creation_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["paths"]["/clients"]["post"]["responses"]["201"]
