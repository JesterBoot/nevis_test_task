from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "sqlite:///./nevis.db"
    db_pool_size: int = Field(default=5, gt=0)
    db_max_overflow: int = Field(default=10, ge=0)
    db_pool_timeout: int = Field(default=30, gt=0)
    db_pool_recycle: int = Field(default=1_800, gt=0)
    db_pool_pre_ping: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, gt=0)
    model_cache_dir: Path = Path("./model-cache")
    model_ready_file: Path | None = None

    max_document_chars: int = Field(default=50_000, gt=0)
    max_chunks: int = Field(default=100, gt=0)
    chunk_size: int = Field(default=1_000, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    search_limit_default: int = Field(default=10, gt=0)
    search_limit_max: int = Field(default=50, gt=0)
    fts_boost: float = Field(default=0.10, ge=0)

    hf_hub_offline: bool = False
    transformers_offline: bool = False

    @model_validator(mode="after")
    def validate_limits(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.search_limit_default > self.search_limit_max:
            raise ValueError("search_limit_default must not exceed search_limit_max")
        if self.chunk_size > self.max_document_chars:
            raise ValueError("chunk_size must not exceed max_document_chars")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
