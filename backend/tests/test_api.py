"""Tests for the FastAPI RAG endpoint."""

from fastapi.testclient import TestClient

from backend.app.main import app, get_hr_agent, get_ingestion_service, get_rag_service
from backend.app.services.rag_service import RAGAnswer
from backend.app.services.rag_service import RAGService


class FakeRAGService:
    """Capture endpoint input and return a deterministic answer."""

    def __init__(self) -> None:
        self.received_question: str | None = None

    def answer_question(self, question: str, n_results: int = 3) -> RAGAnswer:
        self.received_question = question
        return RAGAnswer(
            answer="Employees are entitled to 21 annual leave days.",
            sources=[
                {"source_document": "leave_policy.txt", "chunk_index": 0}
            ],
        )


class FakeRetrievalService:
    """Return deterministic source metadata without using embeddings or ChromaDB."""

    def retrieve(self, query: str, n_results: int = 3) -> list[dict[str, object]]:
        return [
            {
                "text": "Employees are entitled to 21 annual leave days.",
                "source_document": "leave_policy.txt",
                "chunk_index": 2,
                "distance": 0.1,
            }
        ]


class FakeLLMService:
    """Generate a deterministic answer without making a Gemini request."""

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: list[dict[str, object]],
    ) -> str:
        return "Employees are entitled to 21 annual leave days."


class FakeIngestionService:
    """Capture document indexing without loading models or writing to ChromaDB."""

    def __init__(self) -> None:
        self.received_filename: str | None = None

    def ingest_document(self, filename: str) -> dict[str, object]:
        self.received_filename = filename
        return {"source_filename": filename, "chunk_count": 1}


class FakeAgent:
    """Capture agent questions without invoking Gemini or other services."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.received_question: str | None = None

    def answer_question(self, question: str) -> str:
        self.received_question = question
        return self.answer


def test_health_endpoint_returns_ok() -> None:
    """The health endpoint reports that the API is available."""
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_agent_endpoint_allows_local_frontend_preflight() -> None:
    """The API permits the Vite development origin to call the agent route."""
    response = TestClient(app).options(
        "/api/agent/ask",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_ask_endpoint_forwards_question_and_returns_answer() -> None:
    """A valid request reaches RAGService and returns the generated answer."""
    fake_service = FakeRAGService()
    app.dependency_overrides[get_rag_service] = lambda: fake_service
    client = TestClient(app)

    try:
        response = client.post(
            "/api/rag/ask",
            json={"question": "How many annual leave days are employees entitled to?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_service.received_question == (
        "How many annual leave days are employees entitled to?"
    )
    assert response.json() == {
        "question": "How many annual leave days are employees entitled to?",
        "answer": "Employees are entitled to 21 annual leave days.",
        "sources": [
            {"source_document": "leave_policy.txt", "chunk_index": 0}
        ],
    }


def test_ask_endpoint_rejects_missing_or_empty_question() -> None:
    """Missing and whitespace-only questions receive validation errors."""
    client = TestClient(app)

    assert client.post("/api/rag/ask", json={}).status_code == 422
    assert client.post("/api/rag/ask", json={"question": "   "}).status_code == 422


def test_ask_endpoint_returns_sources_from_injected_rag_pipeline() -> None:
    """The API exposes source metadata from fake retrieval and LLM services."""
    service = RAGService(FakeRetrievalService(), FakeLLMService())
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = TestClient(app).post(
            "/api/rag/ask",
            json={"question": "How many annual leave days are employees entitled to?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"] == "Employees are entitled to 21 annual leave days."
    assert response.json()["sources"] == [
        {"source_document": "leave_policy.txt", "chunk_index": 2}
    ]


def test_ingest_endpoint_indexes_filename_with_injected_service() -> None:
    """The ingest endpoint delegates indexing and returns its summary."""
    fake_service = FakeIngestionService()
    app.dependency_overrides[get_ingestion_service] = lambda: fake_service

    try:
        response = TestClient(app).post(
            "/api/rag/ingest",
            json={"filename": "leave_policy.txt"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_service.received_filename == "leave_policy.txt"
    assert response.json() == {"filename": "leave_policy.txt", "chunk_count": 1}


def test_ingest_endpoint_rejects_missing_or_empty_filename() -> None:
    """Missing and whitespace-only filenames receive validation errors."""
    client = TestClient(app)

    assert client.post("/api/rag/ingest", json={}).status_code == 422
    assert client.post("/api/rag/ingest", json={"filename": "   "}).status_code == 422


def test_agent_endpoint_routes_employee_question_to_injected_agent() -> None:
    """The agent endpoint returns an injected employee-tool answer."""
    fake_agent = FakeAgent("Aisha Khan (E001) has 21 annual leave days.")
    app.dependency_overrides[get_hr_agent] = lambda: fake_agent

    try:
        response = TestClient(app).post(
            "/api/agent/ask",
            json={"question": "How much leave does E001 have?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_agent.received_question == "How much leave does E001 have?"
    assert response.json() == {
        "question": "How much leave does E001 have?",
        "answer": "Aisha Khan (E001) has 21 annual leave days.",
    }


def test_agent_endpoint_routes_policy_question_to_injected_agent() -> None:
    """Policy questions are delegated to the injected deterministic agent."""
    fake_agent = FakeAgent("Annual leave policy answer")
    app.dependency_overrides[get_hr_agent] = lambda: fake_agent

    try:
        response = TestClient(app).post(
            "/api/agent/ask",
            json={"question": "How many annual leave days are employees entitled to?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert fake_agent.received_question == (
        "How many annual leave days are employees entitled to?"
    )
    assert response.json()["answer"] == "Annual leave policy answer"


def test_agent_endpoint_rejects_missing_or_empty_question() -> None:
    """The agent endpoint rejects missing and whitespace-only questions."""
    client = TestClient(app)

    assert client.post("/api/agent/ask", json={}).status_code == 422
    assert client.post("/api/agent/ask", json={"question": " "}).status_code == 422
