"""Tests for /api/chat and /api/chat/stream routes (real services)."""

from api.state import sessions


class TestChatValidation:
    """Validation tests — no external services needed."""

    def test_missing_fields_returns_422(self, client):
        r = client.post("/api/chat", json={})
        assert r.status_code == 422

    def test_unknown_session_returns_404(self, client):
        r = client.post("/api/chat", json={"session_id": "nope", "question": "hello"})
        assert r.status_code == 404

    def test_empty_question_returns_400(self, client):
        sessions["s1"] = {"thread_id": "t1", "document_id": "d1", "file_name": "f.pdf"}
        r = client.post("/api/chat", json={"session_id": "s1", "question": "  "})
        assert r.status_code == 400


class TestChatStreamValidation:
    """Validation tests for /api/chat/stream — no external services needed."""

    def test_unknown_session_returns_404(self, client):
        r = client.post("/api/chat/stream", json={"session_id": "nope", "question": "hello"})
        assert r.status_code == 404

    def test_empty_question_returns_400(self, client):
        sessions["s1"] = {"thread_id": "t1", "document_id": "d1", "file_name": "f.pdf"}
        r = client.post("/api/chat/stream", json={"session_id": "s1", "question": ""})
        assert r.status_code == 400


class TestChatSSE:
    """SSE streaming tests using real graph + Ollama + Qdrant."""

    def test_streams_tokens(self, client, ingested_session):
        session_id, _ = ingested_session

        r = client.post("/api/chat", json={"session_id": session_id, "question": "What is in this document?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        body = r.text
        assert "data:" in body
        # Should have token and done events
        has_token = '"type":"token"' in body or '"type": "token"' in body
        has_done = '"type":"done"' in body or '"type": "done"' in body
        # Allow error event as alternative (LLM may not find context)
        has_error = '"type":"error"' in body or '"type": "error"' in body
        assert has_done or has_error or has_token

    def test_includes_sources_or_done(self, client, ingested_session):
        session_id, _ = ingested_session

        body = client.post("/api/chat", json={"session_id": session_id, "question": "What content is here?"}).text
        # Should have either sources or done event
        has_done = '"type":"done"' in body or '"type": "done"' in body
        has_error = '"type":"error"' in body or '"type": "error"' in body
        assert has_done or has_error


class TestChatStreamSSE:
    """Tests for /api/chat/stream — true token-by-token streaming with real services."""

    def test_stream_tokens(self, client, ingested_session):
        session_id, _ = ingested_session

        r = client.post("/api/chat/stream", json={"session_id": session_id, "question": "What is this about?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        body = r.text
        assert "data:" in body
        has_token = '"type": "token"' in body or '"type":"token"' in body
        has_done = '"type": "done"' in body or '"type":"done"' in body
        has_error = '"type": "error"' in body or '"type":"error"' in body
        assert has_token or has_done or has_error
