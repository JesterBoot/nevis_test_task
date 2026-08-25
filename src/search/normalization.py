import re

from search.query import NormalizedSearchQuery


def normalize_search_query(value: str) -> NormalizedSearchQuery:
    raw = value.strip()
    normalized = " ".join(raw.lower().split())
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    domain = _extract_domain(normalized)
    return NormalizedSearchQuery(
        raw=raw,
        normalized=normalized,
        compact=compact,
        domain=domain,
    )


def candidate_limit(limit: int) -> int:
    return min(max(limit * 5, 20), 50)


def _extract_domain(value: str) -> str | None:
    candidate = value.rsplit("@", 1)[-1]
    if (
        (candidate != value or "." in candidate)
        and re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", candidate)
    ):
        return candidate
    return None
