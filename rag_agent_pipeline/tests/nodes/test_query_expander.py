"""Tests for nodes.query_expander — multi-query generation (real Ollama)."""

import sys

import nodes.query_expander
_mod = sys.modules["nodes.query_expander"]


class TestExpandQuery:
    """Tests for the expand_query() node function using real LLM."""

    def test_generates_variants(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_EXPANSION_ENABLED", True)

        result = _mod.expand_query(base_state(question="what about the price?"))

        queries = result["expanded_queries"]
        assert len(queries) >= 2
        assert queries[0] == "what about the price?"  # original first

    def test_includes_original_query(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_EXPANSION_ENABLED", True)

        result = _mod.expand_query(base_state(question="my question"))

        assert result["expanded_queries"][0] == "my question"

    def test_fallback_on_llm_error(self, base_state, monkeypatch):
        from langchain_openai import ChatOpenAI

        monkeypatch.setattr(_mod, "QUERY_EXPANSION_ENABLED", True)
        bad_llm = ChatOpenAI(
            model="nonexistent",
            openai_api_key="bad",
            openai_api_base="http://localhost:1/v1",
            max_retries=0,
            timeout=2,
        )
        monkeypatch.setattr(_mod, "LLM", bad_llm)

        result = _mod.expand_query(base_state(question="my question"))

        assert result["expanded_queries"] == ["my question"]

    def test_disabled_returns_single_query(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_EXPANSION_ENABLED", False)

        result = _mod.expand_query(base_state(question="my question"))

        assert result["expanded_queries"] == ["my question"]

    def test_respects_count_limit(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_EXPANSION_ENABLED", True)
        monkeypatch.setattr(_mod, "QUERY_EXPANSION_COUNT", 2)

        result = _mod.expand_query(base_state(question="what is the pricing model?"))

        # original + max 2 variants = 3
        assert len(result["expanded_queries"]) <= 3

    def test_filters_short_variants(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "QUERY_EXPANSION_ENABLED", True)

        result = _mod.expand_query(base_state(question="test query about documents"))

        for q in result["expanded_queries"]:
            assert len(q) >= 5
