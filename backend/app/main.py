"""Application entry point."""

from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.app.services.embedding_service import EmbeddingService
from backend.app.services.ingestion_service import IngestionService, IndexedDocument
from backend.app.services.llm_service import LLMService
from backend.app.services.rag_service import RAGAnswer, RAGService
from backend.app.services.retrieval_service import RetrievalService
from backend.app.services.vector_store_service import VectorStoreService

SAMPLE_DOCUMENT = "leave_policy.txt"
INDEX_MARKER = Path("chroma_db") / ".leave_policy_indexed"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Index the sample HR document once when the application starts."""
    if not INDEX_MARKER.exists():
        get_ingestion_service().ingest_document(SAMPLE_DOCUMENT)
        INDEX_MARKER.parent.mkdir(parents=True, exist_ok=True)
        INDEX_MARKER.touch()
    yield

app = FastAPI(
    title="HRMS AI RAG API",
    description="AI-powered HR Knowledge Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


class RAGQuestionRequest(BaseModel):
    """Request body for asking a question about HR documents."""

    question: str = Field(min_length=1)


class IngestRequest(BaseModel):
    """Request body for indexing an HR document."""

    filename: str = Field(min_length=1)


class RAGSourceResponse(BaseModel):
    """Source document and chunk used to ground an answer."""

    source_document: str
    chunk_index: int


class RAGQuestionResponse(BaseModel):
    """Response body containing the answer and its source metadata."""

    question: str
    answer: str
    sources: list[RAGSourceResponse]


class IngestResponse(BaseModel):
    """Summary of an indexed document."""

    filename: str
    chunk_count: int


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Build and cache the application-level RAG service."""
    embedding_service = EmbeddingService()
    vector_store_service = VectorStoreService()
    retrieval_service = RetrievalService(embedding_service, vector_store_service)
    llm_service = LLMService()
    return RAGService(retrieval_service, llm_service)


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    """Build and cache the application-level ingestion service."""
    return IngestionService()


@app.get("/")
def root():
    return {
        "message": "HRMS AI RAG API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/api/rag/ask", response_model=RAGQuestionResponse)
def ask_rag_question(
    request: RAGQuestionRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGQuestionResponse:
    """Answer a question using the configured RAG pipeline."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    result: RAGAnswer = rag_service.answer_question(question)
    return RAGQuestionResponse(
        question=question,
        answer=result.answer,
        sources=[
            RAGSourceResponse(
                source_document=source["source_document"],
                chunk_index=source["chunk_index"],
            )
            for source in result.sources
        ],
    )


@app.post("/api/rag/ingest", response_model=IngestResponse)
def ingest_document(
    request: IngestRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Index a document through the configured ingestion service."""
    filename = request.filename.strip()
    if not filename:
        raise HTTPException(status_code=422, detail="filename must not be empty")

    result: IndexedDocument = ingestion_service.ingest_document(filename)
    return IngestResponse(
        filename=result["source_filename"],
        chunk_count=result["chunk_count"],
    )