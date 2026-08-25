from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class TextChunk:
    position: int
    content: str


@dataclass(frozen=True)
class EmbeddedChunk:
    position: int
    content: str
    embedding: list[float]


@dataclass(frozen=True)
class ClientSearchMatch:
    id: UUID
    first_name: str
    last_name: str
    email: str


@dataclass(frozen=True)
class DocumentSearchMatch:
    id: UUID
    client_id: UUID
    title: str
    snippet: str
    ranking_score: float


@dataclass
class DocumentRankingCandidate:
    id: UUID
    client_id: UUID
    title: str
    snippet: str
    best_raw_cosine: float
    lexical_match: bool
