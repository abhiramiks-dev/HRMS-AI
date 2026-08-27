"""Tests for document loading and chunking utilities."""

from backend.app.services.document_service import (
    load_text_document,
    split_text_into_chunks,
)


def test_load_text_document() -> None:
    """The sample leave policy can be loaded from the documents directory."""
    document = load_text_document("leave_policy.txt")

    assert "HRMS Demo Company - Leave Policy" in document


def test_split_text_into_multiple_chunks() -> None:
    """Long text is split into multiple chunks of the configured size."""
    chunks = split_text_into_chunks("abcdefghij", chunk_size=4, chunk_overlap=0)

    assert chunks == ["abcd", "efgh", "ij"]


def test_split_text_preserves_overlap() -> None:
    """Adjacent chunks share the configured number of characters."""
    chunks = split_text_into_chunks("abcdefghi", chunk_size=5, chunk_overlap=2)

    assert chunks == ["abcde", "defgh", "ghi"]
    assert chunks[0][-2:] == chunks[1][:2]
    assert chunks[1][-2:] == chunks[2][:2]
