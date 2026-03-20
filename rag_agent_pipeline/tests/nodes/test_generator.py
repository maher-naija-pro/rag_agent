"""Tests for nodes.generator — LLM answer generation.

Default mode: LLM is auto-mocked (returns "Mocked answer from LLM.").
With --llm: runs against real LLM API.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.llm
class TestGenerate:
    def test_returns_answer(self, base_state, sample_context):
        from nodes.generator import generate

        result = generate(base_state(
            question="What is the capital of France?",
            context=sample_context,
            messages=[HumanMessage(content="What is the capital of France?")],
        ))

        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert isinstance(result["messages"][0], AIMessage)

    def test_returns_ai_message(self, base_state, sample_context):
        from nodes.generator import generate

        result = generate(base_state(
            question="What country is Paris in?",
            context=sample_context,
            messages=[HumanMessage(content="What country is Paris in?")],
        ))

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert len(result["messages"][0].content) > 0

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

    def test_llm_error_returns_error_message(self, base_state, sample_context, monkeypatch):
        """When LLM streaming fails, generator returns an error answer."""
        from nodes.generator import generate
        from langchain_openai import ChatOpenAI

        bad_llm = ChatOpenAI(
            model="nonexistent",
            openai_api_key="bad",
            openai_api_base="http://localhost:1/v1",
            max_retries=0,
            timeout=2,
        )
        import nodes.generator as gen_mod
        monkeypatch.setattr(gen_mod, "LLM", bad_llm)

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
        assert "[page 1" in result and "Hello" in result
        assert "[page 2" in result and "World" in result

    def test_includes_line_numbers_single_line(self):
        """When line_start == line_end, format shows 'ligne X'."""
        from nodes.generator import _format_docs

        docs = [Document(page_content="Hello", metadata={"page": 3, "line_start": 7, "line_end": 7})]
        result = _format_docs(docs)
        assert "[page 3, ligne 7]" in result

    def test_includes_line_range(self):
        """When line_start != line_end, format shows 'lignes X-Y'."""
        from nodes.generator import _format_docs

        docs = [Document(page_content="Multi\nline", metadata={"page": 2, "line_start": 5, "line_end": 8})]
        result = _format_docs(docs)
        assert "[page 2, lignes 5-8]" in result

    def test_no_line_metadata_falls_back_to_page_only(self):
        """Without line_start metadata, format shows only page."""
        from nodes.generator import _format_docs

        docs = [Document(page_content="Text", metadata={"page": 4})]
        result = _format_docs(docs)
        assert "[page 4]" in result
        assert "ligne" not in result

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
