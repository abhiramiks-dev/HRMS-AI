"""Query retrieval orchestration for the HRMS-AI RAG pipeline."""

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.vector_store_service import (
    RetrievedChunk,
    VectorStoreService,
)


class RetrievalService:
    """Convert user queries to embeddings and retrieve relevant chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store_service: VectorStoreService,
    ) -> None:
        """Configure the services used for query retrieval.

        Args:
            embedding_service: Service used to embed the user's query.
            vector_store_service: Service used to search stored document chunks.
        """
        self._embedding_service = embedding_service
        self._vector_store_service = vector_store_service

    def retrieve(self, query: str, n_results: int = 3) -> list[RetrievedChunk]:
        """Retrieve the most relevant document chunks for a user query.

        Args:
            query: User question or search text.
            n_results: Maximum number of chunks to retrieve.

        Returns:
            The retrieved chunks returned by the vector store, including
            their text, source document, chunk index, and distance.
        """
        query_embedding = self._embedding_service.embed_query(query)
        return self._vector_store_service.search(query_embedding, n_results)
