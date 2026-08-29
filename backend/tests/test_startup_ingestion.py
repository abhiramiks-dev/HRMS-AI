"""Tests for startup document indexing."""

from pathlib import Path

from fastapi.testclient import TestClient

import backend.app.main as main


class FakeIngestionService:
    """Capture startup ingestion without loading models or databases."""

    def __init__(self) -> None:
        self.received_filename: str | None = None

    def ingest_document(self, filename: str) -> dict[str, object]:
        self.received_filename = filename
        return {"source_filename": filename, "chunk_count": 2}


def test_startup_indexes_sample_document_once(monkeypatch, tmp_path: Path) -> None:
    """Application startup ingests the sample document and uses its marker."""
    fake_service = FakeIngestionService()
    marker = tmp_path / "chroma" / ".indexed"
    monkeypatch.setattr(main, "get_ingestion_service", lambda: fake_service)
    monkeypatch.setattr(main, "INDEX_MARKER", marker)

    with TestClient(main.app):
        assert fake_service.received_filename == "leave_policy.txt"

    assert marker.exists()

    fake_service.received_filename = None
    with TestClient(main.app):
        assert fake_service.received_filename is None
