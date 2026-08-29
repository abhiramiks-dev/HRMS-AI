"""Tool for answering questions from the existing HR policy RAG pipeline."""

from typing import Protocol


class RAGAnswerProtocol(Protocol):
    """Minimal answer shape required from the RAG service."""

    answer: str


class RAGServiceProtocol(Protocol):
    """Interface required by the policy search tool."""

    def answer_question(self, question: str) -> RAGAnswerProtocol:
        """Answer a question using retrieved HR policy context."""


class PolicySearchTool:
    """Delegate HR policy questions to the existing RAG service."""

    def __init__(self, rag_service: RAGServiceProtocol) -> None:
        """Configure the tool with an injected RAG service."""
        self._rag_service = rag_service

    def answer(self, question: str) -> str:
        """Return the generated answer for an HR policy question."""
        return self._rag_service.answer_question(question).answer
