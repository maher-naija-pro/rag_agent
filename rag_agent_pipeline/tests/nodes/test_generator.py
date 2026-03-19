"""Tests for nodes.generator — LLM answer generation."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


class TestGenerate:
    @patch("nodes.generator.LLM")
    def test_returns_answer(self, mock_llm, base_state, sample_context):
        from nodes.generator import generate

        tok1, tok2 = MagicMock(content="Paris "), MagicMock(content="is the capital.")
        mock_llm.stream.return_value = [tok1, tok2]

        result = generate(base_state(
            question="What is the capital of France?",
            context=sample_context,
            messages=[HumanMessage(content="What is the capital of France?")],
        ))

        assert result["answer"] == "Paris is the capital."
        assert isinstance(result["messages"][0], AIMessage)

    @patch("nodes.generator.LLM")
    def test_returns_ai_message(self, mock_llm, base_state, sample_context):
        from nodes.generator import generate

        mock_llm.stream.return_value = [MagicMock(content="Answer.")]

        result = generate(base_state(
            question="test",
            context=sample_context,
            messages=[HumanMessage(content="test")],
        ))

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Answer."

    def test_empty_context_returns_refusal(self, base_state):
        """When no context docs are retrieved, generator refuses instead of calling LLM."""
        from nodes.generator import generate, NO_CONTEXT_ANSWER

        result = generate(base_state(
            question="test",
            context=[],
            messages=[HumanMessage(content="test")],
        ))
        assert result["answer"] == NO_CONTEXT_ANSWER
        assert isinstance(result["messages"][0], AIMessage)

    @patch("nodes.generator.LLM")
    def test_llm_error_returns_error_message(self, mock_llm, base_state, sample_context):
        """When LLM streaming fails, generator returns an error answer."""
        from nodes.generator import generate

        mock_llm.stream.side_effect = RuntimeError("API timeout")

        result = generate(base_state(
            question="test",
            context=sample_context,
            messages=[HumanMessage(content="test")],
        ))

        assert "error" in result["answer"].lower() or "try again" in result["answer"].lower()
        assert isinstance(result["messages"][0], AIMessage)


class TestFormatDocs:
    def test_includes_page_numbers(self):
        from nodes.generator import _format_docs

        docs = [
            Document(page_content="Hello", metadata={"page": 1}),
            Document(page_content="World", metadata={"page": 2}),
        ]
        result = _format_docs(docs)
        assert "[page 1]" in result and "Hello" in result
        assert "[page 2]" in result and "World" in result

    def test_fallback_page(self):
        from nodes.generator import _format_docs

        docs = [Document(page_content="No metadata", metadata={})]
        assert "[page ?]" in _format_docs(docs)

    def test_empty_docs(self):
        from nodes.generator import _format_docs

        assert _format_docs([]) == ""

    def test_separator_between_docs(self):
        from nodes.generator import _format_docs

        docs = [
            Document(page_content="A", metadata={"page": 1}),
            Document(page_content="B", metadata={"page": 2}),
        ]
        result = _format_docs(docs)
        assert "---" in result
