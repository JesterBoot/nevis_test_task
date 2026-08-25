import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import Settings, get_settings
from search.chunking import chunk_text
from search.types import EmbeddedChunk


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        pass


class _BatchEncoder(Protocol):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        pass


class MiniLMEmbeddingProvider:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: _BatchEncoder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._model = model or self._load_model()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        _validate_texts(values)
        if not values:
            return []

        raw_embeddings = self._model.encode(
            values,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return _normalize_embeddings(
            raw_embeddings,
            expected_count=len(values),
            expected_dimension=self.settings.embedding_dimension,
        )

    def _load_model(self) -> SentenceTransformer:
        return SentenceTransformer(
            self.settings.embedding_model,
            cache_folder=str(self.settings.model_cache_dir),
            device="cpu",
            local_files_only=True,
        )


class FakeEmbeddingProvider:
    def __init__(
        self,
        dimension: int | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        resolved_dimension = (
            resolved_settings.embedding_dimension
            if dimension is None
            else dimension
        )
        if resolved_dimension <= 0:
            raise ValueError("dimension must be greater than zero")
        self.dimension = resolved_dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        _validate_texts(values)
        return [self._embed_one(text) for text in values]

    def _embed_one(self, text: str) -> list[float]:
        raw_values: list[int] = []
        block_index = 0
        while len(raw_values) < self.dimension:
            digest = hashlib.sha256(f"{text}\0{block_index}".encode()).digest()
            raw_values.extend(
                int.from_bytes(digest[offset : offset + 4], "big")
                for offset in range(0, len(digest), 4)
            )
            block_index += 1

        vector = (
            np.asarray(raw_values[: self.dimension], dtype=np.float64) / 2**32 * 2 - 1
        )
        norm = float(np.linalg.norm(vector))
        normalized = vector / norm
        return [float(value) for value in normalized]


def embed_document(
    content: str,
    provider: EmbeddingProvider,
    *,
    max_document_chars: int = 50_000,
    max_chunks: int = 100,
    chunk_size: int = 1_000,
    chunk_overlap: int = 100,
) -> list[EmbeddedChunk]:
    chunks = chunk_text(
        content,
        max_document_chars=max_document_chars,
        max_chunks=max_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = provider.embed([chunk.content for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise ValueError("embedding provider returned an invalid batch length")
    return [
        EmbeddedChunk(
            position=chunk.position,
            content=chunk.content,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]


def _validate_texts(texts: Sequence[str]) -> None:
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("all texts must be strings")


def _normalize_embeddings(
    raw_embeddings: object,
    *,
    expected_count: int,
    expected_dimension: int,
) -> list[list[float]]:
    embeddings = np.asarray(raw_embeddings, dtype=np.float64)
    if embeddings.ndim == 1 and expected_count == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.ndim != 2:
        raise ValueError("embedding provider returned an invalid batch shape")
    if embeddings.shape[0] != expected_count:
        raise ValueError("embedding provider returned an invalid batch length")
    if embeddings.shape[1] != expected_dimension:
        raise ValueError(
            "embedding provider returned vectors with dimension "
            f"{embeddings.shape[1]}, expected {expected_dimension}"
        )

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0):
        raise ValueError("embedding provider returned a zero vector")
    normalized = embeddings / norms[:, np.newaxis]
    return [[float(value) for value in vector] for vector in normalized]
