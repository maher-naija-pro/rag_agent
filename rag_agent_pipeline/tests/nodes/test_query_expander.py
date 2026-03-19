"""Tests for nodes.query_expander — multi-query generation."""

from unittest.mock import MagicMock, patch


class TestExpandQuery:
    """Tests for the expand_query() node function.

    Mocks: LLM (external API).
    Never mocks: expand_query logic (our code under test).
    """

    @patch("nodes.query_expander.QUERY_EXPANSION_ENABLED", True)
    @patch("nodes.query_expander.LLM")
    def test_generates_variants(self, mock_llm, base_state):
        from nodes.query_expander import expand_query

        mock_llm.invoke.return_value = MagicMock(
            content="What are the associated costs?\nWhat is the pricing model?\nHow much does it cost?"
        )

        result = expand_query(base_state(question="what about the price?"))

        queries = result["expanded_queries"]
        assert len(queries) >= 2
        assert queries[0] == "what about the price?"  # original first

    @patch("nodes.query_expander.QUERY_EXPANSION_ENABLED", True)
    @patch("nodes.query_expander.LLM")
    def test_includes_original_query(self, mock_llm, base_state):
        from nodes.query_expander import expand_query

        mock_llm.invoke.return_value = MagicMock(content="Variant one\nVariant two")

        result = expand_query(base_state(question="my question"))

        assert result["expanded_queries"][0] == "my question"

    @patch("nodes.query_expander.QUERY_EXPANSION_ENABLED", True)
    @patch("nodes.query_expander.LLM")
    def test_deduplicates_original(self, mock_llm, base_state):
        from nodes.query_expander import expand_query

        mock_llm.invoke.return_value = MagicMock(content="my question\nA variant")

        result = expand_query(base_state(question="my question"))

        # Original should appear only once
        assert result["expanded_queries"].count("my question") == 1

    @patch("nodes.query_expander.QUERY_EXPANSION_ENABLED", True)
    @patch("nodes.query_expander.LLM")
    def test_filters_short_variants(self, mock_llm, base_state):
        from nodes.query_expander import expand_query

        mock_llm.invoke.return_value = MagicMock(content="Good variant here\n\nab\n\n")

        result = expand_query(base_state(question="test query"))

        for q in result["expanded_queries"]:
            assert len(q) >= 5

    @patch("nodes.query_expander.QUERY_EXPANSION_ENABLED", True)
    @patch("nodes.query_expander.LLM")
    def test_fallback_on_llm_error(self, mock_llm, base_state):
        from nodes.query_expander import expand_query

        mock_llm.invoke.side_effect = RuntimeError("API down")

        result = expand_query(base_state(question="my question"))

        assert result["expanded_queries"] == ["my question"]

    @patch("nodes.query_expander.QUERY_EXPANSION_ENABLED", False)
    def test_disabled_returns_single_query(self, base_state):
        from nodes.query_expander import expand_query

        result = expand_query(base_state(question="my question"))

        assert result["expanded_queries"] == ["my question"]

    @patch("nodes.query_expander.QUERY_EXPANSION_ENABLED", True)
    @patch("nodes.query_expander.QUERY_EXPANSION_COUNT", 2)
    @patch("nodes.query_expander.LLM")
    def test_respects_count_limit(self, mock_llm, base_state):
        from nodes.query_expander import expand_query

        mock_llm.invoke.return_value = MagicMock(
            content="Variant one\nVariant two\nVariant three\nVariant four\nVariant five"
        )

        result = expand_query(base_state(question="test"))

        # original + max 2 variants = 3
        assert len(result["expanded_queries"]) <= 3
