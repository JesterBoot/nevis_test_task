from services.documents import _document_chunk_embedding_text


def test_document_embedding_input_combines_title_and_chunk_content() -> None:
    assert (
        _document_chunk_embedding_text(
            "Proof of address",
            "Utility bill issued in August.",
        )
        == "Proof of address\n\nUtility bill issued in August."
    )
