"""Application entry point."""

# from fastapi import FastAPI

# app = FastAPI(title="HRMS-AI-RAG")
from fastapi import FastAPI

app = FastAPI(
    title="HRMS AI RAG API",
    description="AI-powered HR Knowledge Assistant",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "HRMS AI RAG API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }