from collections.abc import AsyncIterator, Sequence
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import delete, select

from core.config import Settings
from db.session import AsyncSession, get_session
from main import create_app
from models import Client, Document, DocumentChunk
from search.dependencies import get_embedding_provider
from tests.support.database import build_test_database_url

DATABASE_URL = build_test_database_url(Settings().database_url)

pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not DATABASE_URL.startswith("postgresql"),
        reason="Lifecycle integration test requires PostgreSQL with pgvector",
    ),
]


class LifecycleEmbeddingProvider:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.strip().lower()
            if normalized == "address proof" or "utility bill" in normalized:
                vectors.append(_unit_vector(0))
            else:
                vectors.append(_unit_vector(1))
        return vectors


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * Settings().embedding_dimension
    vector[index] = 1.0
    return vector


@pytest_asyncio.fixture
async def lifecycle_app(
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
    application.dependency_overrides[get_embedding_provider] = (
        lambda: LifecycleEmbeddingProvider()
    )
    yield application
    application.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_core_client_document_search_lifecycle(
    lifecycle_app: FastAPI,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with TestClient(lifecycle_app) as api_client:
        client_response = api_client.post(
            "/clients",
            json={
                "first_name": "Anton",
                "last_name": "Batiaev",
                "email": "anton.batiaev@neviswealth.com",
                "countryOfResidence": "GB",
            },
        )
        assert client_response.status_code == 201
        client_id = client_response.json()["id"]

        document_response = api_client.post(
            f"/clients/{client_id}/documents",
            json={
                "title": "Proof of address",
                "content": "Utility bill issued in August.",
            },
        )
        assert document_response.status_code == 201
        document_id = document_response.json()["id"]
        document_uuid = UUID(document_id)

        company_search = api_client.get(
            "/search",
            params={"q": "Nevis Wealth"},
        )
        address_search = api_client.get(
            "/search",
            params={"q": "address proof"},
        )
        unrelated_search = api_client.get(
            "/search",
            params={"q": "portfolio allocation"},
        )

    async with session_factory() as session:
        chunks = (
            await session.exec(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_uuid)
                .order_by(DocumentChunk.position)
            )
        ).all()

    assert chunks
    assert chunks[0].position == 0
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding) == Settings().embedding_dimension

    assert company_search.status_code == 200
    company_results = company_search.json()
    assert isinstance(company_results, list)
    assert company_results == [
        {
            "type": "client",
            "id": client_id,
            "first_name": "Anton",
            "last_name": "Batiaev",
            "email": "anton.batiaev@neviswealth.com",
        }
    ]

    assert address_search.status_code == 200
    address_results = address_search.json()
    document_results = [
        result
        for result in address_results
        if result["type"] == "document"
    ]
    assert document_results == [
        {
            "type": "document",
            "id": document_id,
            "client_id": client_id,
            "title": "Proof of address",
            "snippet": "Utility bill issued in August.",
        }
    ]
    assert all("score" not in result for result in address_results)
    assert all("ranking_score" not in result for result in address_results)
    assert all("embedding" not in result for result in address_results)

    assert unrelated_search.status_code == 200
    unrelated_results = unrelated_search.json()
    assert [
        result
        for result in unrelated_results
        if result["type"] == "document"
    ] == []
