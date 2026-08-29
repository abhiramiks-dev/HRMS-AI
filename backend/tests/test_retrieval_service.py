"""Tests for query retrieval orchestration."""

from backend.app.services.retrieval_service import RetrievalService


class FakeEmbeddingService:
    """Capture a query and return a deterministic embedding."""

    def __init__(self) -> None:
        self.received_query: str | None = None

    def embed_query(self, query: str) -> list[float]:
        self.received_query = query
        return [0.25, 0.75]


class FakeVectorStoreService:
    """Capture search arguments and return predefined chunks."""

    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.received_embedding: list[float] | None = None
        self.received_n_results: int | None = None

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3,
    ) -> list[dict[str, object]]:
        self.received_embedding = query_embedding
        self.received_n_results = n_results
        return self.results


def test_retrieve_embeds_query_passes_arguments_and_preserves_results() -> None:
    """Retrieval forwards the query embedding and returns results unchanged."""
    expected_results = [
        {
            "text": "Annual leave is available to employees.",
            "source_document": "leave_policy.txt",
            "chunk_index": 2,
            "distance": 0.14,
        },
        {
            "text": "Requests require manager approval.",
            "source_document": "leave_policy.txt",
            "chunk_index": 3,
            "distance": 0.27,
        },
    ]
    embedding_service = FakeEmbeddingService()
    vector_store_service = FakeVectorStoreService(expected_results)
    service = RetrievalService(embedding_service, vector_store_service)

    results = service.retrieve("How do I request annual leave?", n_results=2)

    assert embedding_service.received_query == "How do I request annual leave?"
    assert vector_store_service.received_embedding == [0.25, 0.75]
    assert vector_store_service.received_n_results == 2
    assert results is expected_results
