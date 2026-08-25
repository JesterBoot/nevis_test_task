from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import create_engine, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from core.config import Settings, get_settings
from db.session import AsyncSession, build_async_engine
from models import Client, Document, DocumentChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(override=False)
DATABASE_URL = Settings().database_url


def _sync_database_url() -> str:
    return DATABASE_URL


def postgres_available() -> bool:
    if not DATABASE_URL.startswith("postgresql"):
        return False
    engine = create_engine(_sync_database_url(), pool_pre_ping=True)
    try:
        with engine.connect():
            return True
    except SQLAlchemyError:
        return False
    finally:
        engine.dispose()


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del config
    if postgres_available():
        return

    skip_database = pytest.mark.skip(
        reason="PostgreSQL/pgvector is unavailable",
    )
    for item in items:
        if item.get_closest_marker("database") is not None:
            item.add_marker(skip_database)


def _alembic_config() -> Config:
    config = Config(str(PROJECT_ROOT / "src" / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))
    return config


@pytest.fixture(scope="session", autouse=True)
def migrated_schema() -> Iterator[None]:
    if not postgres_available():
        yield
        return

    get_settings.cache_clear()
    config = _alembic_config()
    command.upgrade(config, "head")
    yield
    command.downgrade(config, "base")
    get_settings.cache_clear()


@pytest_asyncio.fixture(scope="session")
async def async_engine(migrated_schema: None) -> AsyncIterator[AsyncEngine]:
    del migrated_schema
    settings = Settings(_env_file=None, database_url_raw=DATABASE_URL)
    engine = build_async_engine(settings)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="session")
def session_factory(
    async_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest_asyncio.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as database_session:
        await database_session.exec(delete(DocumentChunk))
        await database_session.exec(delete(Document))
        await database_session.exec(delete(Client))
        await database_session.commit()
        yield database_session
        await database_session.rollback()
