"""Orchestration for the complete HRMS-AI retrieval-augmented generation flow."""

from dataclasses import dataclass
from typing import TypedDict

from backend.app.services.llm_service import LLMService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.vector_store_service import RetrievedChunk


class RAGSource(TypedDict):
    """Public metadata identifying a retrieved source chunk."""

    source_document: str
    chunk_index: int


@dataclass(frozen=True)
class RAGAnswer:
    """Generated answer and the source chunks used as context."""

    answer: str
    sources: list[RAGSource]


class RAGService:
    """Coordinate retrieval and answer generation for a user question."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
    ) -> None:
        """Configure the retrieval and language-model services.

        Args:
            retrieval_service: Service that finds relevant document chunks.
            llm_service: Service that generates a grounded answer.
        """
        self._retrieval_service = retrieval_service
        self._llm_service = llm_service

    def answer_question(self, question: str, n_results: int = 3) -> RAGAnswer:
        """Retrieve context and generate an answer for a user question.

        Args:
            question: User's question.
            n_results: Maximum number of document chunks to retrieve.

        Returns:
            The generated answer and metadata for the retrieved source chunks.
        """
        retrieved_chunks: list[RetrievedChunk] = self._retrieval_service.retrieve(
            question,
            n_results,
        )
        return RAGAnswer(
            answer=self._llm_service.generate_answer(question, retrieved_chunks),
            sources=[
                {
                    "source_document": chunk["source_document"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in retrieved_chunks
            ],
        )
