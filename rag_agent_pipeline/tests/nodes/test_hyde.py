"""Tests for nodes.hyde — hypothetical document embedding.

Default mode: LLM is auto-mocked.  With --llm: real LLM API.
"""

import sys

import pytest

import nodes.hyde
_mod = sys.modules["nodes.hyde"]


@pytest.mark.llm
class TestHyde:
    def test_appends_hypothetical_to_queries(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "HYDE_ENABLED", True)

        result = _mod.hyde(base_state(
            question="What are the payment terms?",
            expanded_queries=["What are the payment terms?"],
        ))

        queries = result["expanded_queries"]
        assert len(queries) == 2
        assert queries[0] == "What are the payment terms?"
        assert len(queries[1]) > 5

    def test_preserves_existing_expanded_queries(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "HYDE_ENABLED", True)

        result = _mod.hyde(base_state(
            question="test question about documents",
            expanded_queries=["query1", "query2"],
        ))

        queries = result["expanded_queries"]
        assert queries[0] == "query1"
        assert queries[1] == "query2"
        assert len(queries) == 3

    def test_disabled_returns_empty(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "HYDE_ENABLED", False)
        result = _mod.hyde(base_state(question="test"))
        assert result == {}

    def test_fallback_on_llm_error(self, base_state, monkeypatch):
        from langchain_openai import ChatOpenAI

        monkeypatch.setattr(_mod, "HYDE_ENABLED", True)
        bad_llm = ChatOpenAI(
            model="nonexistent", openai_api_key="bad",
            openai_api_base="http://localhost:1/v1",
            max_retries=0, timeout=2,
        )
        monkeypatch.setattr(_mod, "LLM", bad_llm)

        result = _mod.hyde(base_state(question="test", expanded_queries=["test"]))
        assert result == {}

    def test_accepts_sufficient_response(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "HYDE_ENABLED", True)

        result = _mod.hyde(base_state(
            question="What is the company revenue?",
            expanded_queries=["What is the company revenue?"],
        ))
        assert "expanded_queries" in result
        assert len(result["expanded_queries"]) == 2

    def test_short_response_skipped(self, base_state, monkeypatch):
        """When the LLM returns a very short passage (<10 chars), hyde skips it."""
        from unittest.mock import MagicMock

        monkeypatch.setattr(_mod, "HYDE_ENABLED", True)
        # Mock LLM to return a too-short response
        short_llm = MagicMock()
        short_llm.invoke.return_value = MagicMock(content="Short")
        monkeypatch.setattr(_mod, "LLM", short_llm)

        result = _mod.hyde(base_state(
            question="test question",
            expanded_queries=["test question"],
        ))
        # Should return empty dict (skip), not update expanded_queries
        assert result == {}
