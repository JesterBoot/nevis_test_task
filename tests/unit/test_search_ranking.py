from uuid import UUID

from services.search import rank_document_candidates

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000002")
CLIENT_ID = UUID("10000000-0000-0000-0000-000000000001")
FIRST_CHUNK_ID = UUID("20000000-0000-0000-0000-000000000001")
SECOND_CHUNK_ID = UUID("20000000-0000-0000-0000-000000000002")


def _row(
    document_id: UUID,
    similarity: float,
    lexical_match: bool,
    *,
    content: str = "Utility bill issued in August.",
    position: int = 0,
    chunk_id: UUID = FIRST_CHUNK_ID,
) -> tuple[UUID, UUID, str, str, float, bool, int, UUID]:
    return (
        document_id,
        CLIENT_ID,
        "Proof of address",
        content,
        similarity,
        lexical_match,
        position,
        chunk_id,
    )


def test_similarity_below_threshold_is_excluded_even_with_fts_match() -> None:
    matches = rank_document_candidates(
        [_row(DOCUMENT_ID, 0.25, True)],
        threshold=0.30,
        fts_boost=0.10,
        snippet_length=240,
    )

    assert matches == []


def test_threshold_boundary_includes_exact_threshold_and_excludes_lower_value() -> None:
    matches = rank_document_candidates(
        [
            _row(DOCUMENT_ID, 0.300, False),
            _row(OTHER_DOCUMENT_ID, 0.299, False),
        ],
        threshold=0.30,
        fts_boost=0.10,
        snippet_length=240,
    )

    assert [match.id for match in matches] == [DOCUMENT_ID]
    assert matches[0].ranking_score == 0.300


def test_document_aggregation_keeps_best_chunk_and_applies_boost_once() -> None:
    matches = rank_document_candidates(
        [
            _row(
                DOCUMENT_ID,
                0.45,
                True,
                content="Less relevant chunk.",
                position=1,
                chunk_id=SECOND_CHUNK_ID,
            ),
            _row(
                DOCUMENT_ID,
                0.51,
                False,
                content="Best semantic chunk.",
            ),
        ],
        threshold=0.30,
        fts_boost=0.10,
        snippet_length=240,
    )

    assert len(matches) == 1
    assert matches[0].ranking_score == 0.61
    assert matches[0].snippet == "Best semantic chunk."


def test_equal_similarity_uses_position_then_chunk_id_for_snippet() -> None:
    matches = rank_document_candidates(
        [
            _row(
                DOCUMENT_ID,
                0.50,
                False,
                content="Position one.",
                position=1,
                chunk_id=SECOND_CHUNK_ID,
            ),
            _row(
                DOCUMENT_ID,
                0.50,
                False,
                content="Position zero.",
                position=0,
                chunk_id=FIRST_CHUNK_ID,
            ),
        ],
        threshold=0.30,
        fts_boost=0.10,
        snippet_length=240,
    )

    assert len(matches) == 1
    assert matches[0].snippet == "Position zero."


def test_document_order_is_deterministic_for_equal_scores() -> None:
    matches = rank_document_candidates(
        [
            _row(OTHER_DOCUMENT_ID, 0.50, False),
            _row(DOCUMENT_ID, 0.50, False),
        ],
        threshold=0.30,
        fts_boost=0.10,
        snippet_length=240,
    )

    assert [match.id for match in matches] == [
        DOCUMENT_ID,
        OTHER_DOCUMENT_ID,
    ]
