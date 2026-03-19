"""Tests for nodes.self_query — metadata filter extraction."""

import sys
from unittest.mock import MagicMock, patch

# Force module import (not the re-exported function from nodes/__init__)
import nodes.self_query
_mod = sys.modules["nodes.self_query"]


class TestSelfQuery:
    """Tests for the self_query() node function.

    Mocks: LLM (external API).
    Never mocks: self_query logic, _parse_filter_response (our code under test).
    """

    @patch.object(_mod, "SELF_QUERY_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_extracts_page_filter(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content='{"page": 5}')
        result = _mod.self_query(base_state(question="What's on page 5?"))
        assert result["metadata_filter"] == {"page": 5}

    @patch.object(_mod, "SELF_QUERY_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_extracts_language_filter(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content='{"language": "fr"}')
        result = _mod.self_query(base_state(question="Résume les sections françaises"))
        assert result["metadata_filter"] == {"language": "fr"}

    @patch.object(_mod, "SELF_QUERY_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_extracts_multiple_filters(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content='{"page": 3, "language": "en"}')
        result = _mod.self_query(base_state(question="Page 3 in English"))
        assert result["metadata_filter"]["page"] == 3
        assert result["metadata_filter"]["language"] == "en"

    @patch.object(_mod, "SELF_QUERY_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_extracts_has_tables(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content='{"has_tables": true}')
        result = _mod.self_query(base_state(question="Show me the tables"))
        assert result["metadata_filter"] == {"has_tables": True}

    @patch.object(_mod, "SELF_QUERY_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_no_filters_returns_empty(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content="{}")
        result = _mod.self_query(base_state(question="What is the main topic?"))
        assert result["metadata_filter"] == {}

    @patch.object(_mod, "SELF_QUERY_ENABLED", False)
    def test_disabled_returns_empty(self, base_state):
        result = _mod.self_query(base_state(question="page 5"))
        assert result["metadata_filter"] == {}

    @patch.object(_mod, "SELF_QUERY_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_fallback_on_llm_error(self, mock_llm, base_state):
        mock_llm.invoke.side_effect = RuntimeError("API down")
        result = _mod.self_query(base_state(question="page 5"))
        assert result["metadata_filter"] == {}

    @patch.object(_mod, "SELF_QUERY_ENABLED", True)
    @patch.object(_mod, "LLM")
    def test_handles_markdown_code_fence(self, mock_llm, base_state):
        mock_llm.invoke.return_value = MagicMock(content='```json\n{"page": 2}\n```')
        result = _mod.self_query(base_state(question="page 2"))
        assert result["metadata_filter"] == {"page": 2}


class TestParseFilterResponse:
    """Unit tests for _parse_filter_response helper — no mocks needed."""

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
