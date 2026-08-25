from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedSearchQuery:
    raw: str
    normalized: str
    compact: str
    domain: str | None
