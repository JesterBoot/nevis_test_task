from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    position: int
    content: str


@dataclass(frozen=True)
class EmbeddedChunk:
    position: int
    content: str
    embedding: list[float]
