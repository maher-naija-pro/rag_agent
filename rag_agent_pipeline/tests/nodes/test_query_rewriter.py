"""Tests for nodes.query_rewriter — query reformulation."""

import sys
from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

import nodes.query_rewriter
_mod = sys.modules["nodes.query_rewriter"]


class TestRewriteQuery:
    """Tests for the rewrite_query() node function.

    Mocks: LLM (external API).
    Never mocks: rewrite_query logic (our code under test).
    """

    @patch.object(_mod, "QUERY_REWRITE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_rewrites_question(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content="What is the pricing structure?")

        result = _mod.rewrite_query(base_state(question="what about the price?"))

        assert result["question"] == "What is the pricing structure?"
        assert result["original_question"] == "what about the price?"

    @patch.object(_mod, "QUERY_REWRITE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_preserves_original_question(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content="Rewritten query")

        result = _mod.rewrite_query(base_state(question="my question"))

        assert result["original_question"] == "my question"

    @patch.object(_mod, "QUERY_REWRITE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_fallback_on_empty_rewrite(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content="")

        result = _mod.rewrite_query(base_state(question="original query"))

        assert result["original_question"] == "original query"
        assert "question" not in result

    @patch.object(_mod, "QUERY_REWRITE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_fallback_on_llm_error(self, mock_llm, base_state):
        mock_llm.invoke.side_effect = RuntimeError("API down")

        result = _mod.rewrite_query(base_state(question="my question"))

        assert result["original_question"] == "my question"
        assert "question" not in result

    @patch.object(_mod, "QUERY_REWRITE_ENABLED", False)
    def test_disabled_passes_through(self, base_state):
        result = _mod.rewrite_query(base_state(question="my question"))

        assert result["original_question"] == "my question"
        assert "question" not in result

    @patch.object(_mod, "QUERY_REWRITE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_uses_conversation_history(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content="What is the project deadline?")

        history = [
            HumanMessage(content="Tell me about the project"),
            AIMessage(content="The project is about building a RAG pipeline."),
            HumanMessage(content="and the deadline?"),
        ]

        result = _mod.rewrite_query(base_state(
            question="and the deadline?",
            messages=history,
        ))

        assert result["question"] == "What is the project deadline?"
        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = call_args[-1].content
        assert "Conversation context" in user_msg
