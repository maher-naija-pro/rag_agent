"""Tests for nodes.query_rewriter — query reformulation.

Default mode: LLM is auto-mocked.  With --llm: real Ollama.
"""

import sys

import pytest
from langchain_core.messages import HumanMessage, AIMessage

import nodes.query_rewriter
_mod = sys.modules["nodes.query_rewriter"]


@pytest.mark.llm
class TestRewriteQuery:
    def test_rewrites_question(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_REWRITE_ENABLED", True)

        result = _mod.rewrite_query(base_state(question="what about the price?"))

        assert "question" in result
        assert isinstance(result["question"], str)
        assert len(result["question"]) > 0
        assert result["original_question"] == "what about the price?"

    def test_preserves_original_question(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_REWRITE_ENABLED", True)

        result = _mod.rewrite_query(base_state(question="my question"))
        assert result["original_question"] == "my question"

    def test_disabled_passes_through(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_REWRITE_ENABLED", False)

        result = _mod.rewrite_query(base_state(question="my question"))
        assert result["original_question"] == "my question"
        assert "question" not in result

    def test_fallback_on_llm_error(self, base_state, monkeypatch):
        from langchain_openai import ChatOpenAI

        monkeypatch.setattr(_mod, "QUERY_REWRITE_ENABLED", True)
        bad_llm = ChatOpenAI(
            model="nonexistent", openai_api_key="bad",
            openai_api_base="http://localhost:1/v1",
            max_retries=0, timeout=2,
        )
        monkeypatch.setattr(_mod, "LLM", bad_llm)

        result = _mod.rewrite_query(base_state(question="my question"))
        assert result["original_question"] == "my question"
        assert "question" not in result

    def test_uses_conversation_history(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_REWRITE_ENABLED", True)

        history = [
            HumanMessage(content="Tell me about the project"),
            AIMessage(content="The project is about building a RAG pipeline."),
            HumanMessage(content="and the deadline?"),
        ]

        result = _mod.rewrite_query(base_state(
            question="and the deadline?",
            messages=history,
        ))

        assert "question" in result
        # Both mock and real LLM should return something longer than empty
        assert len(result["question"]) > 0
