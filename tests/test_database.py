from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import create_engine, delete, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import selectinload
from sqlmodel import select

from core.config import Settings, get_settings
from db.session import AsyncSession, build_async_engine
from models import Client, Document, DocumentChunk
from schemas.documents import DocumentCreate
from search.embeddings import FakeEmbeddingProvider
from services.documents import create_document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(override=False)
DATABASE_URL = Settings().database_url

pytestmark = [
    pytest.mark.database,
    pytest.mark.skipif(
        not DATABASE_URL.startswith("postgresql"),
        reason="B1 database tests require a PostgreSQL/pgvector DATABASE_URL",
    ),
]


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "src" / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))
    return config


@pytest.fixture(scope="session", autouse=True)
def migrated_schema() -> Iterator[None]:
    get_settings.cache_clear()
    config = _alembic_config()
    command.upgrade(config, "head")
    yield
    get_settings.cache_clear()
    command.downgrade(config, "base")


@pytest_asyncio.fixture(scope="session")
async def async_engine() -> AsyncIterator[AsyncEngine]:
    settings = Settings(_env_file=None, database_url=DATABASE_URL)
    engine = build_async_engine(settings)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def session(async_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as database_session:
        await database_session.exec(delete(DocumentChunk))
        await database_session.exec(delete(Document))
        await database_session.exec(delete(Client))
        await database_session.commit()
        yield database_session
        await database_session.rollback()


def _client(
    *,
    email: str = "anton.batiaev@neviswealth.com",
    normalized_email: str = "anton.batiaev@neviswealth.com",
) -> Client:
    return Client(
        first_name="Anton",
        last_name="Batiaev",
        email=email,
        normalized_email=normalized_email,
        email_domain="neviswealth.com",
        email_domain_label="neviswealth",
        country_of_residence="ME",
    )


def _document(client_id) -> Document:
    return Document(
        client_id=client_id,
        title="Proof of address",
        content="Utility bill issued in August.",
    )


def _chunk(document_id, position: int = 0) -> DocumentChunk:
    return DocumentChunk(
        document_id=document_id,
        position=position,
        content="Utility bill issued in August.",
        embedding=[0.0] * 384,
    )


async def test_migration_creates_tables_and_vector_column(
    async_engine: AsyncEngine,
) -> None:
    async with async_engine.connect() as connection:
        tables, columns = await connection.run_sync(_schema_snapshot)

    assert {"clients", "document_chunks", "documents"} <= set(tables)
    assert columns["embedding"].lower() == "vector(384)"


def _schema_snapshot(connection) -> tuple[list[str], dict[str, str]]:
    inspector = inspect(connection)
    tables = inspector.get_table_names()
    columns = {
        column["name"]: str(column["type"])
        for column in inspector.get_columns("document_chunks")
    }
    return tables, columns


async def test_vector_extension_accepts_orm_insert(
    session: AsyncSession,
) -> None:
    client = _client()
    document = _document(client.id)
    chunk = _chunk(document.id)
    session.add(client)
    session.add(document)
    session.add(chunk)

    await session.commit()

    persisted_chunk = await session.get(DocumentChunk, chunk.id)
    assert persisted_chunk is not None
    assert len(persisted_chunk.embedding) == 384


async def test_duplicate_normalized_email_is_rejected(
    session: AsyncSession,
) -> None:
    session.add(_client())
    await session.commit()

    session.add(
        _client(
            email="different.local.part@neviswealth.com",
            normalized_email="anton.batiaev@neviswealth.com",
        )
    )

    with pytest.raises(IntegrityError):
        await session.commit()


async def test_sqlmodel_relationships_persist_and_load(
    session: AsyncSession,
) -> None:
    client = _client()
    document = _document(client.id)
    document.chunks.append(_chunk(document.id))
    client.documents.append(document)

    session.add(client)
    await session.commit()

    loaded_client = (
        await session.exec(
            select(Client)
            .where(Client.id == client.id)
            .options(selectinload(Client.documents))
        )
    ).one()
    loaded_document = (
        await session.exec(
            select(Document)
            .where(Document.id == document.id)
            .options(selectinload(Document.chunks))
        )
    ).one()

    assert len(loaded_client.documents) == 1
    assert loaded_client.documents[0].id == document.id
    assert len(loaded_document.chunks) == 1
    assert loaded_document.chunks[0].position == 0
    assert len(loaded_document.chunks[0].embedding) == 384


async def test_duplicate_chunk_positions_are_rejected(
    session: AsyncSession,
) -> None:
    client = _client()
    document = _document(client.id)
    session.add(client)
    session.add(document)
    await session.commit()

    session.add_all(
        [
            _chunk(document.id, position=0),
            _chunk(document.id, position=0),
        ]
    )

    with pytest.raises(IntegrityError):
        await session.commit()


async def test_document_delete_cascades_to_chunks(
    session: AsyncSession,
) -> None:
    client = _client()
    document = _document(client.id)
    session.add(client)
    session.add(document)
    await session.commit()

    chunk = _chunk(document.id)
    session.add(chunk)
    await session.commit()

    await session.exec(delete(Document).where(Document.id == document.id))
    await session.commit()

    remaining_chunks = (
        await session.exec(
            select(DocumentChunk).where(DocumentChunk.id == chunk.id)
        )
    ).all()
    assert remaining_chunks == []


async def test_document_ingestion_persists_batched_vector_chunks(
    session: AsyncSession,
) -> None:
    client = _client()
    session.add(client)
    await session.commit()

    document = await create_document(
        session,
        client.id,
        DocumentCreate(
            title="Long proof of address",
            content="x" * 2_000,
        ),
        FakeEmbeddingProvider(),
    )

    persisted_chunks = (
        await session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
    ).all()

    assert len(persisted_chunks) == 3
    assert all(len(chunk.embedding) == 384 for chunk in persisted_chunks)


async def test_document_ingestion_rolls_back_after_persistence_failure(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
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
                content="Utility bill issued in August.",
            ),
            FakeEmbeddingProvider(),
        )

    documents = (await session.exec(select(Document))).all()
    chunks = (await session.exec(select(DocumentChunk))).all()

    assert documents == []
    assert chunks == []


def test_migration_downgrade_and_reupgrade() -> None:
    config = _alembic_config()
    command.downgrade(config, "base")

    sync_engine = create_engine(DATABASE_URL)
    try:
        with sync_engine.connect() as connection:
            tables = set(inspect(connection).get_table_names())
        assert not tables.intersection(
            {"clients", "documents", "document_chunks"}
        )
    finally:
        sync_engine.dispose()

    command.upgrade(config, "head")


def test_database_url_is_postgresql_for_this_module() -> None:
    assert DATABASE_URL.startswith("postgresql")
