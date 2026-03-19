"""Tests for /api/ingest routes."""

from unittest.mock import patch

from langchain_core.documents import Document

from api.state import sessions, jobs


class TestIngestValidation:
    """Validation tests — no mocks needed."""

    def test_no_file_returns_422(self, client):
        r = client.post("/api/ingest")
        assert r.status_code == 422

    def test_wrong_extension_returns_400(self, client):
        r = client.post(
            "/api/ingest",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )
        assert r.status_code == 400

    def test_wrong_mime_returns_400(self, client):
        r = client.post(
            "/api/ingest",
            files={"file": ("test.pdf", b"not a pdf", "text/plain")},
        )
        assert r.status_code == 400

    @patch("api.routes.ingest.MAX_FILE_SIZE", 10)  # 10 bytes
    def test_file_too_large_returns_413(self, client, sample_pdf_bytes):
        r = client.post(
            "/api/ingest",
            files={"file": ("big.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 413

    @patch("api.routes.ingest.graph")
    def test_valid_pdf_accepted(self, mock_graph, client, sample_pdf_bytes):
        mock_graph.invoke.return_value = {
            "raw_pages": [Document(page_content="page 1", metadata={"page": 1})],
            "chunks": [
                Document(page_content="chunk 1", metadata={"page": 1}),
                Document(page_content="chunk 2", metadata={"page": 1}),
            ],
        }

        r = client.post(
            "/api/ingest",
            files={"file": ("doc.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 202
        data = r.json()
        assert data["status"] == "ready"
        assert "session_id" in data
        assert "job_id" in data
        assert data["file_name"] == "doc.pdf"

    @patch("api.routes.ingest.graph")
    def test_ingest_creates_session(self, mock_graph, client, sample_pdf_bytes):
        mock_graph.invoke.return_value = {
            "raw_pages": [],
            "chunks": [],
        }

        r = client.post(
            "/api/ingest",
            files={"file": ("doc.pdf", sample_pdf_bytes, "application/pdf")},
        )
        session_id = r.json()["session_id"]
        assert session_id in sessions

    @patch("api.routes.ingest.graph")
    def test_ingest_failure_returns_500(self, mock_graph, client, sample_pdf_bytes):
        mock_graph.invoke.side_effect = RuntimeError("LLM error")

        r = client.post(
            "/api/ingest",
            files={"file": ("doc.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 500


class TestIngestStatus:
    def test_not_found(self, client):
        r = client.get("/api/ingest/nonexistent")
        assert r.status_code == 404

    def test_returns_job(self, client):
        jobs["j1"] = {"status": "ready", "progress": 100, "session_id": "s1",
                       "document_id": "d1", "file_name": "test.pdf"}

        data = client.get("/api/ingest/j1").json()
        assert data["status"] == "ready"
        assert data["progress"] == 100
