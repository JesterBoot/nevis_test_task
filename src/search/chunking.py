from search.types import TextChunk


def chunk_text(
    content: str,
    *,
    max_document_chars: int = 50_000,
    max_chunks: int = 100,
    chunk_size: int = 1_000,
    chunk_overlap: int = 100,
) -> list[TextChunk]:
    #  Split text into fixed-size overlapping windows.
    _validate_configuration(
        max_document_chars=max_document_chars,
        max_chunks=max_chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if len(content) > max_document_chars:
        raise ValueError(
            "content exceeds the maximum document size of "
            f"{max_document_chars} characters"
        )

    if len(content) <= chunk_size:
        chunks = [TextChunk(position=0, content=content)]
    else:
        step = chunk_size - chunk_overlap
        chunks = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunks.append(
                TextChunk(
                    position=len(chunks),
                    content=content[start:end],
                )
            )
            if end == len(content):
                break
            start += step

    if len(chunks) > max_chunks:
        raise ValueError(
            "content produces "
            f"{len(chunks)} chunks, exceeding the maximum of {max_chunks}"
        )
    return chunks


def _validate_configuration(
    *,
    max_document_chars: int,
    max_chunks: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    if max_document_chars <= 0:
        raise ValueError("max_document_chars must be greater than zero")
    if max_chunks <= 0:
        raise ValueError("max_chunks must be greater than zero")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
