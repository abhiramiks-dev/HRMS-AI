"""Tests for persistent vector storage and similarity search."""

from pathlib import Path

from backend.app.services.vector_store_service import VectorStoreService


def test_chunks_embeddings_and_metadata_are_stored(tmp_path: Path) -> None:
    """Stored chunks can be retrieved with their embeddings and metadata."""
    service = VectorStoreService(persist_directory=tmp_path / "chroma")
    chunks = ["annual leave is 21 days", "sick leave is paid"]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    service.add_document_chunks(chunks, embeddings, "leave_policy.txt")
    results = service.search([1.0, 0.0], n_results=2)

    assert len(results) == 2
    assert results[0]["text"] == chunks[0]
    assert results[0]["source_document"] == "leave_policy.txt"
    assert results[0]["chunk_index"] == 0
    assert isinstance(results[0]["distance"], float)


def test_query_embedding_retrieves_relevant_chunk(tmp_path: Path) -> None:
    """The nearest vector is returned first for a query embedding."""
    service = VectorStoreService(persist_directory=tmp_path / "chroma")

    service.add_document_chunks(
        ["annual leave details", "emergency leave details"],
        [[1.0, 0.0], [0.0, 1.0]],
        "leave_policy.txt",
    )

    results = service.search([0.9, 0.1], n_results=1)

    assert results[0]["text"] == "annual leave details"
