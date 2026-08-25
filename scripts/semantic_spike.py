import sys
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import Settings

QUERY = "address proof"


@dataclass(frozen=True)
class FixtureDocument:
    name: str
    text: str


FIXTURE_DOCUMENTS = (
    FixtureDocument(
        name="utility_bill",
        text="Proof of address\n\nUtility bill issued in August.",
    ),
    FixtureDocument(
        name="passport",
        text="Passport biographical page and identity document.",
    ),
    FixtureDocument(
        name="bank_statement",
        text="Bank statement showing recent account activity.",
    ),
)


@dataclass(frozen=True)
class SimilarityResult:
    name: str
    score: float


def load_embedding_model(settings: Settings) -> SentenceTransformer:
    return SentenceTransformer(
        settings.embedding_model,
        cache_folder=str(settings.model_cache_dir),
        local_files_only=True,
    )


def calculate_similarities(
    model: SentenceTransformer,
    query: str = QUERY,
    documents: tuple[FixtureDocument, ...] = FIXTURE_DOCUMENTS,
) -> list[SimilarityResult]:
    texts = [query, *(document.text for document in documents)]
    embeddings = np.asarray(
        model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    )
    query_embedding = embeddings[0]

    results = [
        SimilarityResult(
            name=document.name,
            score=float(np.dot(query_embedding, document_embedding)),
        )
        for document, document_embedding in zip(
            documents,
            embeddings[1:],
            strict=True,
        )
    ]
    return sorted(results, key=lambda result: (-result.score, result.name))


def print_results(results: list[SimilarityResult]) -> None:
    print(f"query: {QUERY}")
    print("similarities:")
    for result in results:
        print(f"{result.name}: {result.score:.6f}")
    print("ranking:")
    for position, result in enumerate(results, start=1):
        print(f"{position}. {result.name}")


def run_spike(
    settings: Settings | None = None,
) -> list[SimilarityResult]:
    resolved_settings = settings or Settings()
    model = load_embedding_model(resolved_settings)
    return calculate_similarities(model)


def main() -> int:
    try:
        results = run_spike()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Semantic spike failed: {exc}", file=sys.stderr)
        return 1

    print_results(results)
    if not results or results[0].name != "utility_bill":
        print(
            "Semantic acceptance failed: utility_bill did not rank first.",
            file=sys.stderr,
        )
        return 1

    print("semantic acceptance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
