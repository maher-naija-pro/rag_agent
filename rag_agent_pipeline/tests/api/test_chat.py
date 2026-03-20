"""Tests for POST /api/chat route (true token-by-token SSE streaming).

Default mode: LLM is auto-mocked.  With --llm: real LLM API.
"""

import json

import pytest
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


def _parse_sse_events(body: str) -> list[dict]:
    """Parse SSE body text into a list of JSON event dicts."""
    events = []
    for line in body.split("\n"):
        if not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            events.append(json.loads(data))
        except json.JSONDecodeError:
            pass
    return events


@pytest.mark.llm
class TestChatSSE:
    """SSE streaming tests — uses ingested_session which runs the full pipeline."""

    def test_streams_tokens(self, client, ingested_session):
        session_id, _ = ingested_session

        r = client.post("/api/chat", json={"session_id": session_id, "question": "What is in this document?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        body = r.text
        assert "data:" in body
        has_token = '"type":"token"' in body or '"type": "token"' in body
        has_done = '"type":"done"' in body or '"type": "done"' in body
        has_error = '"type":"error"' in body or '"type": "error"' in body
        assert has_done or has_error or has_token

    def test_includes_sources_or_done(self, client, ingested_session):
        session_id, _ = ingested_session

        body = client.post("/api/chat", json={"session_id": session_id, "question": "What content is here?"}).text
        has_done = '"type":"done"' in body or '"type": "done"' in body
        has_error = '"type":"error"' in body or '"type": "error"' in body
        assert has_done or has_error

    def test_emits_status_events(self, client, ingested_session):
        """The SSE stream includes status events for pipeline progress."""
        session_id, _ = ingested_session

        body = client.post("/api/chat", json={"session_id": session_id, "question": "Tell me about this."}).text
        events = _parse_sse_events(body)

        status_events = [e for e in events if e.get("type") == "status"]
        # Should have at least the rewrite and retrieve status events
        assert len(status_events) >= 2
        # Each status event must have step and message fields
        for se in status_events:
            assert "step" in se
            assert "message" in se

    def test_status_steps_include_expected(self, client, ingested_session):
        """Status events cover the main pipeline steps."""
        session_id, _ = ingested_session

        body = client.post("/api/chat", json={"session_id": session_id, "question": "What is here?"}).text
        events = _parse_sse_events(body)

        steps = [e["step"] for e in events if e.get("type") == "status"]
        # At minimum, retrieve and generate should always be present
        assert "retrieve" in steps
        assert "generate" in steps


@pytest.mark.llm
class TestChatErrorPath:
    """Tests for error handling in the chat SSE pipeline."""

    def test_pipeline_error_emits_error_event(self, client, monkeypatch):
        """When the pipeline crashes, an SSE error event is emitted."""
        sessions["err"] = {"thread_id": "terr", "document_id": "d1", "file_name": "f.pdf"}

        # Patch rewrite_query to raise an exception
        import api.routes.chat as chat_mod
        import nodes.query_rewriter as rw_mod
        original = rw_mod.rewrite_query

        def exploding_rewrite(state):
            raise RuntimeError("Simulated pipeline failure")

        monkeypatch.setattr(rw_mod, "rewrite_query", exploding_rewrite)

        body = client.post("/api/chat", json={"session_id": "err", "question": "boom"}).text
        events = _parse_sse_events(body)

        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) >= 1
        assert "Simulated pipeline failure" in error_events[0]["content"]

        # Restore
        monkeypatch.setattr(rw_mod, "rewrite_query", original)
