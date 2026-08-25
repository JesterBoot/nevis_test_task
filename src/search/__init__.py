"""Reusable text preparation and embedding providers."""

from search.chunking import chunk_text
from search.embeddings import (
    EmbeddingProvider,
    FakeEmbeddingProvider,
    MiniLMEmbeddingProvider,
    embed_document,
)
from search.normalization import (
    candidate_limit,
    normalize_search_query,
)
from search.query import NormalizedSearchQuery
from search.types import (
    ClientSearchMatch,
    DocumentRankingCandidate,
    DocumentSearchMatch,
    EmbeddedChunk,
    TextChunk,
)

__all__ = (
    "ClientSearchMatch",
    "DocumentRankingCandidate",
    "DocumentSearchMatch",
    "EmbeddedChunk",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "MiniLMEmbeddingProvider",
    "NormalizedSearchQuery",
    "TextChunk",
    "candidate_limit",
    "chunk_text",
    "embed_document",
    "normalize_search_query",
)
