"""Tests for end-to-end RAG orchestration."""

from backend.app.services.rag_service import RAGService
from backend.app.services.vector_store_service import RetrievedChunk


class FakeRetrievalService:
    """Capture retrieval arguments and return prepared chunks."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.received_question: str | None = None
        self.received_n_results: int | None = None

    def retrieve(self, query: str, n_results: int = 3) -> list[RetrievedChunk]:
        self.received_question = query
        self.received_n_results = n_results
        return self.chunks


class FakeLLMService:
    """Capture generation arguments and return a prepared answer."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.received_question: str | None = None
        self.received_chunks: list[RetrievedChunk] | None = None

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        self.received_question = question
        self.received_chunks = retrieved_chunks
        return self.answer


def test_answer_question_orchestrates_retrieval_and_generation() -> None:
    """The question, result limit, and retrieved chunks flow through unchanged."""
    question = "How many annual leave days are available?"
    retrieved_chunks: list[RetrievedChunk] = [
        {
            "text": "Employees receive 21 annual leave days.",
            "source_document": "leave_policy.txt",
            "chunk_index": 0,
            "distance": 0.08,
        }
    ]
    retrieval_service = FakeRetrievalService(retrieved_chunks)
    llm_service = FakeLLMService("Employees receive 21 annual leave days.")
    service = RAGService(retrieval_service, llm_service)

    result = service.answer_question(question, n_results=5)

    assert retrieval_service.received_question == question
    assert retrieval_service.received_n_results == 5
    assert llm_service.received_question == question
    assert llm_service.received_chunks is retrieved_chunks
    assert result.answer == "Employees receive 21 annual leave days."
    assert result.sources == [
        {"source_document": "leave_policy.txt", "chunk_index": 0}
    ]
