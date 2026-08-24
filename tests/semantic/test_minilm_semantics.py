import os

import pytest

from scripts.semantic_spike import run_spike

pytestmark = pytest.mark.semantic


def test_minilm_ranks_utility_bill_for_address_proof() -> None:
    try:
        results = run_spike()
    except Exception as exc:
        if os.getenv("ALLOW_SEMANTIC_TEST_SKIP") == "1":
            pytest.skip(f"Local MiniLM model is unavailable: {exc}")
        raise AssertionError(
            "The real MiniLM semantic test requires an available local model. "
            "Set ALLOW_SEMANTIC_TEST_SKIP=1 only for local development."
        ) from exc

    assert [result.name for result in results] == [
        "utility_bill",
        "passport",
        "bank_statement",
    ]
    assert results[0].score > results[1].score
    assert results[0].score > results[2].score
