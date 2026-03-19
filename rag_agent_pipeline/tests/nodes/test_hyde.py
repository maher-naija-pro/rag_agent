"""Tests for nodes.hyde — hypothetical document embedding."""

import sys
from unittest.mock import MagicMock, patch

# Force-import the module (not the function re-exported by nodes/__init__)
import nodes.hyde
_mod = sys.modules["nodes.hyde"]


class TestHyde:
    """Tests for the hyde() node function.

    Mocks: LLM (external API).
    Never mocks: hyde logic (our code under test).
    """

    @patch.object(_mod, "HYDE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_appends_hypothetical_to_queries(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(
            content="Payment terms are net 30 days from invoice date. Late fees apply at 1.5% per month."
        )

        result = _mod.hyde(base_state(
            question="What are the payment terms?",
            expanded_queries=["What are the payment terms?"],
        ))

        queries = result["expanded_queries"]
        assert len(queries) == 2
        assert queries[0] == "What are the payment terms?"
        assert "payment" in queries[1].lower()

    @patch.object(_mod, "HYDE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_preserves_existing_expanded_queries(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content="Hypothetical passage here.")

        result = _mod.hyde(base_state(
            question="test",
            expanded_queries=["query1", "query2"],
        ))

        queries = result["expanded_queries"]
        assert queries[0] == "query1"
        assert queries[1] == "query2"
        assert len(queries) == 3

    @patch.object(_mod, "HYDE_ENABLED", False)
    def test_disabled_returns_empty(self, base_state):
        result = _mod.hyde(base_state(question="test"))
        assert result == {}

    @patch.object(_mod, "HYDE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_fallback_on_llm_error(self, mock_llm, base_state):
        mock_llm.invoke.side_effect = RuntimeError("API down")
        result = _mod.hyde(base_state(question="test", expanded_queries=["test"]))
        assert result == {}

    @patch.object(_mod, "HYDE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_skips_on_short_response(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content="Too short")
        result = _mod.hyde(base_state(question="test", expanded_queries=["test"]))
        assert result == {}

    @patch.object(_mod, "HYDE_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_accepts_sufficient_response(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(
            content="This is a sufficiently long hypothetical document passage for testing."
        )
        result = _mod.hyde(base_state(question="test", expanded_queries=["test"]))
        assert "expanded_queries" in result
        assert len(result["expanded_queries"]) == 2
