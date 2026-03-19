"""Tests for /api/ingest routes (real graph + Qdrant)."""

from api.state import sessions, jobs


class TestIngestValidation:
    """Validation tests — no external services needed."""

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

    def test_file_too_large_returns_413(self, client, sample_pdf_bytes, monkeypatch):
        import api.routes.ingest as ingest_mod
        monkeypatch.setattr(ingest_mod, "MAX_FILE_SIZE", 10)  # 10 bytes

        r = client.post(
            "/api/ingest",
            files={"file": ("big.pdf", sample_pdf_bytes, "application/pdf")},
        )
        assert r.status_code == 413

    def test_valid_pdf_accepted(self, client, sample_pdf_bytes, test_collection):
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

    def test_ingest_creates_session(self, client, sample_pdf_bytes, test_collection):
        r = client.post(
            "/api/ingest",
            files={"file": ("doc.pdf", sample_pdf_bytes, "application/pdf")},
        )
        session_id = r.json()["session_id"]
        assert session_id in sessions


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
