"""Tests for document ingestion orchestration."""

from backend.app.services.ingestion_service import IngestionService


class FakeEmbeddingService:
    """Return deterministic embeddings for test chunks."""

    def __init__(self) -> None:
        self.received_chunks: list[str] | None = None

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        self.received_chunks = chunks
        return [[float(index)] for index in range(len(chunks))]


class FakeVectorStoreService:
    """Capture data passed to the vector store."""

    def __init__(self) -> None:
        self.received_chunks: list[str] | None = None
        self.received_embeddings: list[list[float]] | None = None
        self.received_filename: str | None = None

    def add_document_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source_filename: str,
    ) -> None:
        self.received_chunks = chunks
        self.received_embeddings = embeddings
        self.received_filename = source_filename


def test_document_is_loaded_chunked_embedded_and_stored() -> None:
    """Ingestion passes each stage's output to the next stage."""
    embedding_service = FakeEmbeddingService()
    vector_store_service = FakeVectorStoreService()
    loaded_filenames: list[str] = []

    def load_document(filename: str) -> str:
        loaded_filenames.append(filename)
        return "document text"

    def split_document(text: str) -> list[str]:
        assert text == "document text"
        return ["document", "text"]

    service = IngestionService(
        embedding_service=embedding_service,
        vector_store_service=vector_store_service,
        document_loader=load_document,
        text_splitter=split_document,
    )

    result = service.ingest_document("leave_policy.txt")

    assert loaded_filenames == ["leave_policy.txt"]
    assert embedding_service.received_chunks == ["document", "text"]
    assert vector_store_service.received_chunks == ["document", "text"]
    assert vector_store_service.received_embeddings == [[0.0], [1.0]]
    assert vector_store_service.received_filename == "leave_policy.txt"
    assert result == {"source_filename": "leave_policy.txt", "chunk_count": 2}
