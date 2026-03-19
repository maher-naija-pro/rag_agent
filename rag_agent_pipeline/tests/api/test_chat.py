"""Tests for /api/chat and /api/chat/stream routes."""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from api.state import sessions


class TestChatValidation:
    """Validation tests — no mocks needed, tests real route logic."""

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
    """Validation tests for /api/chat/stream — no mocks needed."""

    def test_unknown_session_returns_404(self, client):
        r = client.post("/api/chat/stream", json={"session_id": "nope", "question": "hello"})
        assert r.status_code == 404

    def test_empty_question_returns_400(self, client):
        sessions["s1"] = {"thread_id": "t1", "document_id": "d1", "file_name": "f.pdf"}
        r = client.post("/api/chat/stream", json={"session_id": "s1", "question": ""})
        assert r.status_code == 400


class TestChatSSE:
    """SSE streaming tests.

    Mocks: graph.invoke (LangGraph orchestrator — external infra).
    Uses real Document objects instead of MagicMock for context docs.
    """

    @patch("api.routes.chat.graph")
    def test_streams_tokens(self, mock_graph, client):
        sessions["s1"] = {"thread_id": "t1", "document_id": "d1", "file_name": "f.pdf"}

        mock_graph.invoke.return_value = {
            "context": [Document(page_content="chunk", metadata={"page": 2})],
            "answer": "The answer is 42.",
        }

        r = client.post("/api/chat", json={"session_id": "s1", "question": "What?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        body = r.text
        assert "data:" in body
        assert '"type":"token"' in body or '"type": "token"' in body
        assert '"type":"done"' in body or '"type": "done"' in body

    @patch("api.routes.chat.graph")
    def test_includes_sources(self, mock_graph, client):
        sessions["s1"] = {"thread_id": "t1", "document_id": "d1", "file_name": "f.pdf"}

        mock_graph.invoke.return_value = {
            "context": [
                Document(page_content="chunk A", metadata={"page": 3}),
                Document(page_content="chunk B", metadata={"page": 7}),
            ],
            "answer": "Answer.",
        }

        body = client.post("/api/chat", json={"session_id": "s1", "question": "Q?"}).text
        assert '"type":"sources"' in body or '"type": "sources"' in body

    @patch("api.routes.chat.graph")
    def test_error_event_on_failure(self, mock_graph, client):
        sessions["s1"] = {"thread_id": "t1", "document_id": "d1", "file_name": "f.pdf"}
        mock_graph.invoke.side_effect = RuntimeError("boom")

        body = client.post("/api/chat", json={"session_id": "s1", "question": "Q?"}).text
        assert '"type":"error"' in body or '"type": "error"' in body


class TestChatStreamSSE:
    """Tests for /api/chat/stream — true token-by-token streaming.

    Mocks: LLM, get_store, _build_reranker, rewrite_query, checkpointer, graph
    (all external infra). Tests the actual SSE event assembly logic.
    """

    @patch("api.routes.chat.graph")
    @patch("api.routes.chat.checkpointer")
    @patch("api.routes.chat._build_reranker")
    @patch("api.routes.chat.get_store")
    @patch("api.routes.chat.LLM")
    def test_stream_tokens(self, mock_llm, mock_store, mock_reranker, mock_cp, mock_graph, client):
        sessions["s1"] = {"thread_id": "t1", "document_id": "d1", "file_name": "f.pdf"}

        # Retriever returns candidates
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="chunk", metadata={"page": 2}),
        ]
        mock_store.return_value.as_retriever.return_value = mock_retriever

        # Reranker passes through
        mock_r = MagicMock()
        mock_r.compress_documents.return_value = [
            Document(page_content="chunk", metadata={"page": 2, "relevance_score": 0.9}),
        ]
        mock_reranker.return_value = mock_r

        # Checkpointer returns empty
        mock_cp.get.return_value = None

        # LLM streams tokens
        tok1, tok2 = MagicMock(content="Hello"), MagicMock(content=" world")
        mock_llm.stream.return_value = [tok1, tok2]

        # Graph invoke for saving turn
        mock_graph.invoke.return_value = {}

        r = client.post("/api/chat/stream", json={"session_id": "s1", "question": "What?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        body = r.text
        assert '"type": "token"' in body or '"type":"token"' in body
        assert "Hello" in body
        assert "world" in body
        assert '"type": "done"' in body or '"type":"done"' in body
        assert '"type": "sources"' in body or '"type":"sources"' in body

    @patch("api.routes.chat.graph")
    @patch("api.routes.chat.checkpointer")
    @patch("api.routes.chat._build_reranker")
    @patch("api.routes.chat.get_store")
    @patch("api.routes.chat.LLM")
    def test_stream_error(self, mock_llm, mock_store, mock_reranker, mock_cp, mock_graph, client):
        sessions["s1"] = {"thread_id": "t1", "document_id": "d1", "file_name": "f.pdf"}

        # Make retrieval fail
        mock_store.return_value.as_retriever.side_effect = RuntimeError("Qdrant down")

        r = client.post("/api/chat/stream", json={"session_id": "s1", "question": "Q?"})
        body = r.text
        assert '"type": "error"' in body or '"type":"error"' in body
