import pytest

from search.chunking import TextChunk, chunk_text
from search.embeddings import FakeEmbeddingProvider, embed_document


def test_short_content_returns_one_chunk() -> None:
    assert chunk_text("short") == [TextChunk(position=0, content="short")]


@pytest.mark.parametrize(
    ("content_length", "expected_ranges"),
    [
        (1_000, [(0, 1_000)]),
        (1_001, [(0, 1_000), (900, 1_001)]),
        (2_000, [(0, 1_000), (900, 1_900), (1_800, 2_000)]),
    ],
)
def test_fixed_window_boundaries(
    content_length: int,
    expected_ranges: list[tuple[int, int]],
) -> None:
    content = "".join(str(index % 10) for index in range(content_length))

    chunks = chunk_text(content)

    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.content for chunk in chunks] == [
        content[start:end] for start, end in expected_ranges
    ]


def test_chunking_is_deterministic_and_does_not_split_intelligently() -> None:
    content = "".join(chr(0x1000 + index) for index in range(2_000))

    first_result = chunk_text(content)
    second_result = chunk_text(content)

    assert first_result == second_result
    assert first_result[0].content == content[:1_000]
    assert first_result[1].content == content[900:1_900]
    assert first_result[2].content == content[1_800:2_000]


def test_document_size_is_rejected_before_embedding() -> None:
    provider = _RecordingProvider()

    with pytest.raises(ValueError, match="maximum document size"):
        embed_document(
            "x" * 1_001,
            provider,
            max_document_chars=1_000,
        )

    assert provider.calls == []


def test_chunk_count_is_rejected_before_embedding() -> None:
    provider = _RecordingProvider()

    with pytest.raises(ValueError, match="maximum of 2"):
        embed_document(
            "x" * 2_000,
            provider,
            max_chunks=2,
        )

    assert provider.calls == []


def test_invalid_chunking_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("content", chunk_size=100, chunk_overlap=100)


class _RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return FakeEmbeddingProvider().embed(texts)
