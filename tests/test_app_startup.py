import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings
from app.main import create_app


def test_application_imports_without_database_connection() -> None:
    client = TestClient(create_app())

    response = client.get("/openapi.json")
    docs_response = client.get("/docs")

    assert response.status_code == 200
    assert docs_response.status_code == 200
    assert response.json()["info"]["title"] == "Nevis Backend API"
    assert response.json()["info"]["version"] == "0.1.0"


def test_default_settings_are_bootstrap_safe() -> None:
    settings = Settings(_env_file=None)

    assert settings.database_url == "sqlite:///./nevis.db"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.max_document_chars == 50_000
    assert settings.chunk_size == 1_000
    assert settings.chunk_overlap == 100
    assert settings.search_limit_default == 10
    assert settings.search_limit_max == 50


def test_settings_can_be_overridden_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@db/nevis")
    monkeypatch.setenv("MODEL_CACHE_DIR", "/tmp/nevis-models")
    monkeypatch.setenv("SEARCH_LIMIT_MAX", "25")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+psycopg://user:pass@db/nevis"
    assert str(settings.model_cache_dir) == "/tmp/nevis-models"
    assert settings.search_limit_max == 25
    assert settings.hf_hub_offline is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size", 0),
        ("chunk_overlap", 1000),
        ("search_limit_default", 51),
    ],
)
def test_invalid_numeric_configuration_is_rejected(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
