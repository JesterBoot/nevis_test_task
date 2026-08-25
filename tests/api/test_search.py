from collections.abc import AsyncIterator, Iterator
from uuid import UUID

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlmodel import delete

from core.config import Settings
from db.session import AsyncSession, get_session
from main import create_app
from models import Client, Document, DocumentChunk
from search.dependencies import get_embedding_provider

load_dotenv(override=False)
DATABASE_URL = Settings().database_url

pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not DATABASE_URL.startswith("postgresql"),
        reason="B6 search tests require PostgreSQL with pgvector",
    ),
]


class SearchFixtureEmbeddingProvider:
    dimension = 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if (
                text.strip().lower() == "address proof"
                or "utility bill" in text.lower()
            ):
                vectors.append(_unit_vector(0))
            elif "passport" in text.lower():
                vectors.append(_unit_vector(1))
            elif "bank statement" in text.lower():
                vectors.append(_unit_vector(2))
            else:
                vectors.append(_unit_vector(3))
        return vectors


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * 384
    vector[index] = 1.0
    return vector


def _low_similarity_vector() -> list[float]:
    vector = [0.0] * 384
    vector[0] = 0.25
    vector[1] = (1 - 0.25**2) ** 0.5
    return vector


@pytest_asyncio.fixture
async def search_context(
    async_engine: AsyncEngine,
) -> AsyncIterator[tuple[FastAPI, AsyncSession, SearchFixtureEmbeddingProvider]]:
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    provider = SearchFixtureEmbeddingProvider()

    async with session_factory() as session:
        await session.exec(delete(DocumentChunk))
        await session.exec(delete(Document))
        await session.exec(delete(Client))
        await session.commit()

        async def override_get_session() -> AsyncIterator[AsyncSession]:
            yield session

        application = create_app()
        application.dependency_overrides[get_session] = override_get_session
        application.dependency_overrides[get_embedding_provider] = (
            lambda: provider
        )
        yield application, session, provider
        application.dependency_overrides.clear()
        await session.rollback()


@pytest.fixture
def api_client(
    search_context: tuple[
        FastAPI,
        AsyncSession,
        SearchFixtureEmbeddingProvider,
    ],
) -> Iterator[TestClient]:
    application, _, _ = search_context
    with TestClient(application) as client:
        yield client


async def _persist_client(
    session: AsyncSession,
    *,
    first_name: str = "Anton",
    last_name: str = "Batiaev",
    email: str = "anton.batiaev@neviswealth.com",
) -> Client:
    client = Client(
        first_name=first_name,
        last_name=last_name,
        email=email,
        normalized_email=email,
        email_domain="neviswealth.com",
        email_domain_label="neviswealth",
    )
    session.add(client)
    await session.commit()
    return client


async def _persist_document(
    session: AsyncSession,
    client_id: UUID,
    *,
    title: str,
    content: str,
    embedding: list[float],
    extra_embedding: list[float] | None = None,
) -> Document:
    document = Document(
        client_id=client_id,
        title=title,
        content=content,
    )
    session.add(document)
    await session.flush()
    chunks = [
        DocumentChunk(
            document_id=document.id,
            position=0,
            content=content,
            embedding=embedding,
        )
    ]
    if extra_embedding is not None:
        chunks.append(
            DocumentChunk(
                document_id=document.id,
                position=1,
                content="Address proof lexical phrase.",
                embedding=extra_embedding,
            )
        )
    session.add_all(chunks)
    await session.commit()
    return document


@pytest.mark.asyncio
async def test_search_returns_document_result(
    search_context: tuple[
        FastAPI,
        AsyncSession,
        SearchFixtureEmbeddingProvider,
    ],
) -> None:
    application, session, _ = search_context
    client = await _persist_client(session)
    document = await _persist_document(
        session,
        client.id,
        title="Proof of address",
        content="Utility bill issued in August.",
        embedding=_unit_vector(0),
    )

    with TestClient(application) as api_client:
        response = api_client.get("/search", params={"q": "address proof"})

    assert response.status_code == 200
    body = response.json()
    assert [item["type"] for item in body] == ["document"]
    assert body[0]["id"] == str(document.id)
    assert "score" not in body[0]
    assert "ranking_score" not in body[0]
    assert "embedding" not in body[0]


@pytest.mark.asyncio
async def test_company_exact_domain_and_fts_fallback(
    search_context: tuple[
        FastAPI,
        AsyncSession,
        SearchFixtureEmbeddingProvider,
    ],
) -> None:
    application, session, _ = search_context
    exact_client = await _persist_client(session)
    fts_client = await _persist_client(
        session,
        first_name="Grace",
        last_name="Hopper",
        email="grace@other.example",
    )
    fts_client.email_domain = "other.example"
    fts_client.email_domain_label = "other"
    await session.commit()

    with TestClient(application) as api_client:
        exact_response = api_client.get(
            "/search",
            params={"q": "Nevis Wealth"},
        )
        fts_response = api_client.get(
            "/search",
            params={"q": "Grace Hopper"},
        )

    assert exact_response.status_code == 200
    assert exact_response.json()[0]["id"] == str(exact_client.id)
    assert fts_response.status_code == 200
    assert fts_response.json()[0]["id"] == str(fts_client.id)


@pytest.mark.asyncio
async def test_document_threshold_is_applied_before_fts_boost(
    search_context: tuple[
        FastAPI,
        AsyncSession,
        SearchFixtureEmbeddingProvider,
    ],
) -> None:
    application, session, _ = search_context
    client = await _persist_client(session)
    low_similarity_document = await _persist_document(
        session,
        client.id,
        title="Address proof",
        content="Address proof lexical phrase.",
        embedding=_low_similarity_vector(),
    )

    with TestClient(application) as api_client:
        response = api_client.get(
            "/search",
            params={"q": "address proof"},
        )

    assert response.status_code == 200
    assert all(
        result["id"] != str(low_similarity_document.id)
        for result in response.json()
    )


@pytest.mark.asyncio
async def test_multiple_chunks_are_collapsed_and_boosted_once(
    search_context: tuple[
        FastAPI,
        AsyncSession,
        SearchFixtureEmbeddingProvider,
    ],
) -> None:
    application, session, _ = search_context
    client = await _persist_client(session)
    document = await _persist_document(
        session,
        client.id,
        title="Proof of address",
        content="Utility bill issued in August.",
        embedding=_unit_vector(0),
        extra_embedding=_unit_vector(0),
    )

    with TestClient(application) as api_client:
        response = api_client.get(
            "/search",
            params={"q": "address proof"},
        )

    document_results = [
        result
        for result in response.json()
        if result["type"] == "document"
    ]
    assert [result["id"] for result in document_results] == [str(document.id)]


@pytest.mark.parametrize(
    "params",
    [
        {"q": ""},
        {"q": "   "},
        {"q": "address proof", "limit": "0"},
        {"q": "address proof", "limit": "51"},
        {},
    ],
)
def test_invalid_search_parameters_return_422(
    api_client: TestClient,
    params: dict[str, str],
) -> None:
    response = api_client.get("/search", params=params)

    assert response.status_code == 422


def test_search_openapi_contract(api_client: TestClient) -> None:
    response = api_client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["paths"]["/search"]["get"]
