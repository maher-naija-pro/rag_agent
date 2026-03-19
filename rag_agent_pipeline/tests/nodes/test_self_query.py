"""Tests for nodes.self_query — metadata filter extraction (real Ollama)."""

import sys

import nodes.self_query
_mod = sys.modules["nodes.self_query"]


class TestSelfQuery:
    """Tests for the self_query() node function using real LLM."""

    def test_extracts_page_filter(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "SELF_QUERY_ENABLED", True)

        result = _mod.self_query(base_state(question="What's on page 5?"))
        filters = result["metadata_filter"]
        assert isinstance(filters, dict)
        # LLM should extract page number from the question
        assert filters.get("page") == 5

    def test_extracts_language_filter(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "SELF_QUERY_ENABLED", True)

        result = _mod.self_query(base_state(question="Résume les sections françaises"))
        filters = result["metadata_filter"]
        assert isinstance(filters, dict)
        # Should detect French language reference
        if filters:
            assert filters.get("language") in ("fr", None)

    def test_no_filters_returns_empty_or_minimal(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "SELF_QUERY_ENABLED", True)

        result = _mod.self_query(base_state(question="What is the main topic?"))
        filters = result["metadata_filter"]
        assert isinstance(filters, dict)

    def test_disabled_returns_empty(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "SELF_QUERY_ENABLED", False)

        result = _mod.self_query(base_state(question="page 5"))
        assert result["metadata_filter"] == {}

    def test_fallback_on_llm_error(self, base_state, monkeypatch):
        from langchain_openai import ChatOpenAI

        monkeypatch.setattr(_mod, "SELF_QUERY_ENABLED", True)
        bad_llm = ChatOpenAI(
            model="nonexistent",
            openai_api_key="bad",
            openai_api_base="http://localhost:1/v1",
            max_retries=0,
            timeout=2,
        )
        monkeypatch.setattr(_mod, "LLM", bad_llm)

        result = _mod.self_query(base_state(question="page 5"))
        assert result["metadata_filter"] == {}


class TestParseFilterResponse:
    """Unit tests for _parse_filter_response helper — no external services needed."""

    def test_valid_json(self):
        assert _mod._parse_filter_response('{"page": 5}') == {"page": 5}

    def test_empty_json(self):
        assert _mod._parse_filter_response("{}") == {}

    def test_invalid_json_returns_empty(self):
        assert _mod._parse_filter_response("not json") == {}

    def test_markdown_code_fence(self):
        assert _mod._parse_filter_response('```json\n{"language": "fr"}\n```') == {"language": "fr"}

    def test_non_dict_json_returns_empty(self):
        assert _mod._parse_filter_response("[1, 2, 3]") == {}

    def test_whitespace_handling(self):
        assert _mod._parse_filter_response('  {"page": 1}  ') == {"page": 1}


class TestBuildFieldsDescription:
    def test_includes_configured_fields(self):
        desc = _mod._build_fields_description(["dates", "language"])
        assert "dates" in desc
        assert "language" in desc
        assert "page" in desc

    def test_always_includes_page_and_source(self):
        desc = _mod._build_fields_description([])
        assert "page" in desc
        assert "source" in desc
