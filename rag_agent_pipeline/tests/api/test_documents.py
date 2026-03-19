"""Tests for /api/documents routes."""

from api.state import sessions


class TestDocuments:
    def test_empty_list(self, client):
        data = client.get("/api/documents").json()
        assert data["documents"] == []

    def test_list_after_session_added(self, client):
        sessions["s1"] = {
            "thread_id": "t1",
            "document_id": "d1",
            "file_name": "test.pdf",
            "pages": 5,
            "chunks": 20,
        }
        data = client.get("/api/documents").json()
        assert len(data["documents"]) == 1
        assert data["documents"][0]["name"] == "test.pdf"

    def test_delete_not_found(self, client):
        r = client.delete("/api/documents/nonexistent")
        assert r.status_code == 404

    def test_delete_existing(self, client, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")

        sessions["s1"] = {
            "thread_id": "t1",
            "document_id": "d1",
            "file_name": "test.pdf",
            "file_path": str(pdf),
            "pages": 1,
            "chunks": 3,
        }

        r = client.delete("/api/documents/d1")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"
        assert "s1" not in sessions
        assert not pdf.exists()

    def test_delete_missing_file_still_succeeds(self, client):
        """If the PDF file was already removed from disk, delete still returns 200."""
        sessions["s1"] = {
            "thread_id": "t1",
            "document_id": "d1",
            "file_name": "gone.pdf",
            "file_path": "/tmp/does_not_exist_abc.pdf",
            "pages": 1,
            "chunks": 3,
        }

        r = client.delete("/api/documents/d1")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"
        assert "s1" not in sessions
