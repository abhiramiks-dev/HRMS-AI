"""Orchestration for indexing HRMS documents."""

from collections.abc import Callable
from typing import Protocol, TypedDict

from backend.app.services.document_service import (
    load_text_document,
    split_text_into_chunks,
)
from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store_service import VectorStoreService


class EmbeddingServiceProtocol(Protocol):
    """Interface required from an embedding service."""

    def embed_documents(self, chunks: list[str]) -> list[list[float]]:
        """Create embeddings for document chunks."""


class VectorStoreServiceProtocol(Protocol):
    """Interface required from a vector store service."""

    def add_document_chunks(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source_filename: str,
    ) -> None:
        """Store chunks, embeddings, and source metadata."""


class IndexedDocument(TypedDict):
    """Summary of a document indexing operation."""

    source_filename: str
    chunk_count: int


class IngestionService:
    """Load, chunk, embed, and store a document using existing services."""

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol | None = None,
        vector_store_service: VectorStoreServiceProtocol | None = None,
        document_loader: Callable[[str], str] = load_text_document,
        text_splitter: Callable[[str], list[str]] = split_text_into_chunks,
    ) -> None:
        """Configure the services used during document ingestion.

        Args:
            embedding_service: Service used to generate chunk embeddings.
            vector_store_service: Service used to persist chunks and embeddings.
            document_loader: Function that loads a document by filename.
            text_splitter: Function that splits document text into chunks.
        """
        self._embedding_service = embedding_service or EmbeddingService()
        self._vector_store_service = vector_store_service or VectorStoreService()
        self._document_loader = document_loader
        self._text_splitter = text_splitter

    def ingest_document(self, filename: str) -> IndexedDocument:
        """Index a document and return a summary of the indexed content.

        Args:
            filename: Name of the document to load from the documents directory.

        Returns:
            The source filename and number of chunks indexed.
        """
        text = self._document_loader(filename)
        chunks = self._text_splitter(text)
        embeddings = self._embedding_service.embed_documents(chunks)
        self._vector_store_service.add_document_chunks(chunks, embeddings, filename)

        return {"source_filename": filename, "chunk_count": len(chunks)}
