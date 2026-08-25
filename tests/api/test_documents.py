from collections.abc import AsyncIterator, Iterator
from uuid import uuid4

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
from schemas.documents import DocumentCreate
from search.dependencies import get_embedding_provider
from search.embeddings import FakeEmbeddingProvider
from services.documents import TransactionOwnershipError, create_document
from tests.support.database import build_test_database_url

DATABASE_URL = build_test_database_url(Settings().database_url)
pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not DATABASE_URL.startswith("postgresql"),
        reason="Document API tests require PostgreSQL with pgvector",
    ),
]


class RecordingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        values = list(texts)
        self.calls.append(values)
        return super().embed(values)


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("injected embedding failure")


class TransactionStateEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__()
        self.session = session
        self.transaction_states: list[bool] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.transaction_states.append(self.session.in_transaction())
        return super().embed(texts)


@pytest_asyncio.fixture
async def document_api_context(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[
    tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ]
]:
    async with session_factory() as session:
        await session.exec(delete(DocumentChunk))
        await session.exec(delete(Document))
        await session.exec(delete(Client))
        await session.commit()

    async def override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    provider = RecordingEmbeddingProvider()
    application = create_app()
    application.dependency_overrides[get_session] = override_get_session
    application.dependency_overrides[get_embedding_provider] = lambda: provider
    yield application, session_factory, provider
    application.dependency_overrides.clear()


@pytest.fixture
def api_client(
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
) -> Iterator[TestClient]:
    application, _, _ = document_api_context
    with TestClient(application) as client:
        yield client


def _client_payload(email: str = "anton.batiaev@neviswealth.com") -> dict[str, str]:
    return {
        "first_name": "Anton",
        "last_name": "Batiaev",
        "email": email,
    }


def _create_client(api_client: TestClient) -> str:
    response = api_client.post("/clients", json=_client_payload())
    assert response.status_code == 201
    return response.json()["id"]


def test_create_document_returns_source_fields_without_embeddings(
    api_client: TestClient,
) -> None:
    client_id = _create_client(api_client)
    response = api_client.post(
        f"/clients/{client_id}/documents",
        json={
            "title": "Proof of address",
            "content": "Utility bill issued in August.",
        },
    )

    assert response.status_code == 201
    assert response.json()["client_id"] == client_id
    assert response.json()["title"] == "Proof of address"
    assert response.json()["content"] == "Utility bill issued in August."
    assert "embedding" not in response.json()
    assert "chunks" not in response.json()


def test_unknown_client_returns_404_without_embedding(
    api_client: TestClient,
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
) -> None:
    _, _, provider = document_api_context
    response = api_client.post(
        f"/clients/{uuid4()}/documents",
        json={"title": "Proof of address", "content": "Utility bill."},
    )

    assert response.status_code == 404
    assert provider.calls == []


@pytest.mark.parametrize(
    ("title", "content"),
    [
        ("", "Utility bill."),
        ("   ", "Utility bill."),
        ("Proof of address", ""),
        ("Proof of address", " \n\t "),
    ],
)
def test_empty_or_whitespace_document_fields_return_422(
    api_client: TestClient,
    title: str,
    content: str,
) -> None:
    client_id = _create_client(api_client)
    response = api_client.post(
        f"/clients/{client_id}/documents",
        json={"title": title, "content": content},
    )

    assert response.status_code == 422


def test_document_size_limit_returns_422(
    api_client: TestClient,
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limited_settings = Settings(
        _env_file=None,
        max_document_chars=10,
        chunk_size=10,
        chunk_overlap=1,
    )
    monkeypatch.setattr(
        "services.documents.get_settings",
        lambda: limited_settings,
    )
    client_id = _create_client(api_client)

    response = api_client.post(
        f"/clients/{client_id}/documents",
        json={"title": "Proof", "content": "x" * 11},
    )

    assert response.status_code == 422
    assert document_api_context[2].calls == []


def test_chunk_count_limit_is_rejected_before_embedding(
    api_client: TestClient,
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limited_settings = Settings(
        _env_file=None,
        max_document_chars=2_000,
        max_chunks=2,
        chunk_size=1_000,
        chunk_overlap=100,
    )
    monkeypatch.setattr(
        "services.documents.get_settings",
        lambda: limited_settings,
    )
    client_id = _create_client(api_client)

    response = api_client.post(
        f"/clients/{client_id}/documents",
        json={"title": "Long document", "content": "x" * 2_000},
    )

    assert response.status_code == 422
    assert document_api_context[2].calls == []


def test_all_chunks_are_embedded_in_one_batch(
    api_client: TestClient,
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
) -> None:
    client_id = _create_client(api_client)
    content = "x" * 2_000

    response = api_client.post(
        f"/clients/{client_id}/documents",
        json={"title": "Long document", "content": content},
    )

    assert response.status_code == 201
    assert len(document_api_context[2].calls) == 1
    assert document_api_context[2].calls[0] == [
        f"Long document\n\n{content[:1_000]}",
        f"Long document\n\n{content[900:1_900]}",
        f"Long document\n\n{content[1_800:]}",
    ]


@pytest.mark.asyncio
async def test_persistence_failure_rolls_back_document_and_chunks(
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, session_factory, provider = document_api_context

    async with session_factory() as session:
        client = Client(
            first_name="Anton",
            last_name="Batiaev",
            email="anton@example.com",
            normalized_email="anton@example.com",
            email_domain="example.com",
            email_domain_label="example",
        )
        session.add(client)
        await session.commit()

        async def failing_flush(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected persistence failure")

        monkeypatch.setattr(session, "flush", failing_flush)

        with pytest.raises(RuntimeError, match="injected persistence failure"):
            await create_document(
                session,
                client.id,
                DocumentCreate(
                    title="Proof of address",
                    content="Utility bill.",
                ),
                provider,
            )

    async with session_factory() as verification_session:
        documents = (await verification_session.exec(select(Document))).all()
        chunks = (await verification_session.exec(select(DocumentChunk))).all()

    assert documents == []
    assert chunks == []


@pytest.mark.asyncio
async def test_embedding_runs_after_lookup_transaction_finishes(
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
) -> None:
    _, session_factory, _ = document_api_context

    async with session_factory() as session:
        client = Client(
            first_name="Anton",
            last_name="Batiaev",
            email="transaction-boundary@example.com",
            normalized_email="transaction-boundary@example.com",
            email_domain="example.com",
            email_domain_label="example",
        )
        session.add(client)
        await session.commit()

        provider = TransactionStateEmbeddingProvider(session)
        await create_document(
            session,
            client.id,
            DocumentCreate(
                title="Proof of address",
                content="Utility bill.",
            ),
            provider,
        )

    assert provider.transaction_states == [False]


@pytest.mark.asyncio
async def test_external_transaction_is_not_rolled_back(
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
) -> None:
    _, session_factory, _ = document_api_context

    async with session_factory() as session:
        client = Client(
            first_name="Anton",
            last_name="Batiaev",
            email="external-transaction@example.com",
            normalized_email="external-transaction@example.com",
            email_domain="example.com",
            email_domain_label="example",
        )
        session.add(client)
        await session.commit()

        async with session.begin():
            with pytest.raises(TransactionOwnershipError):
                await create_document(
                    session,
                    client.id,
                    DocumentCreate(
                        title="Proof of address",
                        content="Utility bill.",
                    ),
                    FakeEmbeddingProvider(),
                )

            assert session.in_transaction()


@pytest.mark.asyncio
async def test_embedding_failure_writes_nothing(
    document_api_context: tuple[
        FastAPI,
        async_sessionmaker[AsyncSession],
        RecordingEmbeddingProvider,
    ],
) -> None:
    _, session_factory, _ = document_api_context

    async with session_factory() as session:
        client = Client(
            first_name="Anton",
            last_name="Batiaev",
            email="embedding@example.com",
            normalized_email="embedding@example.com",
            email_domain="example.com",
            email_domain_label="example",
        )
        session.add(client)
        await session.commit()

        with pytest.raises(RuntimeError, match="injected embedding failure"):
            await create_document(
                session,
                client.id,
                DocumentCreate(
                    title="Proof of address",
                    content="Utility bill.",
                ),
                FailingEmbeddingProvider(),
            )

    async with session_factory() as verification_session:
        documents = (await verification_session.exec(select(Document))).all()
        chunks = (await verification_session.exec(select(DocumentChunk))).all()

    assert documents == []
    assert chunks == []


def test_openapi_exposes_document_creation_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/openapi.json")

    assert response.status_code == 200
    assert response.json()["paths"]["/clients/{id}/documents"]["post"]
