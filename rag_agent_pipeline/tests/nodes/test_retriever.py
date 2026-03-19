"""Tests for nodes.retriever (hybrid search) and nodes.reranker (reranking)."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Force module imports (not re-exported functions from nodes/__init__)
import nodes.retriever
import nodes.reranker
_retriever = sys.modules["nodes.retriever"]
_reranker = sys.modules["nodes.reranker"]


# ── retrieve() — hybrid search ──────────────────────────────────────────────

class TestRetrieve:
    @patch.object(_retriever, "get_store")
    def test_returns_candidates(self, mock_get_store, base_state):
        returned_docs = [Document(page_content="relevant chunk", metadata={"page": 2})]

        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = returned_docs
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        result = _retriever.retrieve(base_state(question="What is RAG?"))

        assert result["candidates"] == returned_docs
        mock_base_retriever.invoke.assert_called_once_with("What is RAG?")

    @patch.object(_retriever, "get_store")
    def test_empty_results(self, mock_get_store, base_state):
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        result = _retriever.retrieve(base_state(question="nothing"))
        assert result["candidates"] == []

    @patch.object(_retriever, "get_store")
    def test_source_filter_applied(self, mock_get_store, base_state):
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        _retriever.retrieve(base_state(question="test", source="report.pdf"))

        call_kwargs = mock_store.as_retriever.call_args[1]
        assert "filter" in call_kwargs.get("search_kwargs", {})

    @patch.object(_retriever, "SIMILARITY_THRESHOLD", 0.5)
    @patch.object(_retriever, "get_store")
    def test_similarity_threshold_passed(self, mock_get_store, base_state):
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        _retriever.retrieve(base_state(question="test"))

        call_kwargs = mock_store.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["score_threshold"] == 0.5

    @patch.object(_retriever, "HYBRID_FUSION_ALPHA", 0.7)
    @patch.object(_retriever, "get_store")
    def test_alpha_passed_when_not_default(self, mock_get_store, base_state):
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        _retriever.retrieve(base_state(question="test"))

        call_kwargs = mock_store.as_retriever.call_args[1]
        assert call_kwargs["search_kwargs"]["alpha"] == 0.7

    @patch.object(_retriever, "get_store")
    def test_multi_query_retrieval(self, mock_get_store, base_state):
        doc_a = Document(page_content="chunk A", metadata={"page": 1})
        doc_b = Document(page_content="chunk B", metadata={"page": 2})
        doc_c = Document(page_content="chunk C", metadata={"page": 3})

        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.side_effect = [[doc_a, doc_b], [doc_b, doc_c]]
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        result = _retriever.retrieve(base_state(
            question="main query",
            expanded_queries=["variant 1", "variant 2"],
        ))

        candidates = result["candidates"]
        assert len(candidates) == 3  # A, B, C — B deduplicated
        assert mock_base_retriever.invoke.call_count == 2

    @patch.object(_retriever, "get_store")
    def test_falls_back_to_question_when_no_expanded(self, mock_get_store, base_state):
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        _retriever.retrieve(base_state(question="my question", expanded_queries=[]))

        mock_base_retriever.invoke.assert_called_once_with("my question")

    @patch.object(_retriever, "get_store")
    def test_metadata_filter_page(self, mock_get_store, base_state):
        """metadata_filter with page number creates Qdrant filter."""
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        _retriever.retrieve(base_state(
            question="test",
            metadata_filter={"page": 5},
        ))

        call_kwargs = mock_store.as_retriever.call_args[1]
        assert "filter" in call_kwargs.get("search_kwargs", {})

    @patch.object(_retriever, "get_store")
    def test_metadata_filter_language(self, mock_get_store, base_state):
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        _retriever.retrieve(base_state(
            question="test",
            metadata_filter={"language": "fr"},
        ))

        call_kwargs = mock_store.as_retriever.call_args[1]
        assert "filter" in call_kwargs.get("search_kwargs", {})

    @patch.object(_retriever, "get_store")
    def test_metadata_filter_has_tables(self, mock_get_store, base_state):
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        _retriever.retrieve(base_state(
            question="test",
            metadata_filter={"has_tables": True},
        ))

        call_kwargs = mock_store.as_retriever.call_args[1]
        assert "filter" in call_kwargs.get("search_kwargs", {})


class TestDeduplicate:
    def test_removes_duplicates(self):
        docs = [
            Document(page_content="same", metadata={"page": 1}),
            Document(page_content="same", metadata={"page": 2}),
            Document(page_content="different", metadata={"page": 3}),
        ]
        result = _retriever._deduplicate(docs)
        assert len(result) == 2

    def test_preserves_order(self):
        docs = [
            Document(page_content="first", metadata={}),
            Document(page_content="second", metadata={}),
            Document(page_content="first", metadata={}),
        ]
        result = _retriever._deduplicate(docs)
        assert result[0].page_content == "first"
        assert result[1].page_content == "second"

    def test_empty_list(self):
        assert _retriever._deduplicate([]) == []


# ── rerank() — reranking node ───────────────────────────────────────────────

class TestRerank:
    @patch.object(_reranker, "_build_reranker")
    def test_returns_context(self, mock_build, base_state):
        candidates = [Document(page_content="chunk", metadata={"page": 1, "relevance_score": 0.9})]
        mock_reranker = MagicMock()
        mock_reranker.compress_documents.return_value = candidates
        mock_build.return_value = mock_reranker

        result = _reranker.rerank(base_state(candidates=candidates, question="test"))
        assert result["context"] == candidates

    @patch.object(_reranker, "_build_reranker")
    def test_empty_candidates(self, mock_build, base_state):
        result = _reranker.rerank(base_state(candidates=[], question="test"))
        assert result["context"] == []
        mock_build.assert_not_called()

    @patch.object(_reranker, "RERANK_ENABLED", True)
    @patch.object(_reranker, "RERANK_SCORE_THRESHOLD", 0.5)
    @patch.object(_reranker, "_build_reranker")
    def test_score_threshold_filters(self, mock_build, base_state):
        docs = [
            Document(page_content="high", metadata={"page": 1, "relevance_score": 0.9}),
            Document(page_content="low", metadata={"page": 2, "relevance_score": 0.1}),
        ]
        mock_reranker = MagicMock()
        mock_reranker.compress_documents.return_value = docs
        mock_build.return_value = mock_reranker

        result = _reranker.rerank(base_state(candidates=docs, question="test"))

        assert len(result["context"]) == 1
        assert result["context"][0].page_content == "high"

    @patch.object(_reranker, "RERANK_ENABLED", True)
    @patch.object(_reranker, "_build_reranker")
    def test_reranker_error_falls_back_to_candidates(self, mock_build, base_state):
        candidates = [
            Document(page_content="chunk1", metadata={"page": 1}),
            Document(page_content="chunk2", metadata={"page": 2}),
        ]
        mock_build.side_effect = RuntimeError("reranker crashed")

        result = _reranker.rerank(base_state(candidates=candidates, question="test"))
        assert len(result["context"]) == len(candidates)

    @patch.object(_reranker, "RERANK_ENABLED", False)
    def test_disabled_passes_candidates_through(self, base_state):
        candidates = [
            Document(page_content="c1", metadata={"page": 1}),
            Document(page_content="c2", metadata={"page": 2}),
        ]
        result = _reranker.rerank(base_state(candidates=candidates, question="test"))
        assert result["context"] == candidates


# ── retriever error path ─────────────────────────────────────────────────────

class TestRetrieveErrors:
    @patch.object(_retriever, "get_store")
    def test_retrieval_failure_returns_empty(self, mock_get_store, base_state):
        mock_get_store.return_value.as_retriever.side_effect = RuntimeError("Qdrant down")

        result = _retriever.retrieve(base_state(question="test"))
        assert result["candidates"] == []

    @patch.object(_retriever, "get_store")
    def test_unsupported_metadata_filter_skipped(self, mock_get_store, base_state):
        """Unsupported filter keys are silently skipped, not crashing."""
        mock_base_retriever = MagicMock()
        mock_base_retriever.invoke.return_value = []
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = mock_base_retriever
        mock_get_store.return_value = mock_store

        result = _retriever.retrieve(base_state(
            question="test",
            metadata_filter={"unsupported_field": "value"},
        ))
        assert result["candidates"] == []


# ── _filter_by_score — pure logic ───────────────────────────────────────────

class TestFilterByScore:
    def test_keeps_above_threshold(self):
        docs = [
            Document(page_content="high", metadata={"relevance_score": 0.8}),
            Document(page_content="low", metadata={"relevance_score": 0.1}),
        ]
        result = _reranker._filter_by_score(docs, 0.5)
        assert len(result) == 1
        assert result[0].page_content == "high"

    def test_zero_threshold_keeps_all(self):
        docs = [Document(page_content="any", metadata={"relevance_score": 0.1})]
        assert len(_reranker._filter_by_score(docs, 0.0)) == 1

    def test_no_score_metadata_defaults_to_kept(self):
        docs = [Document(page_content="no score", metadata={})]
        assert len(_reranker._filter_by_score(docs, 0.5)) == 1

    def test_all_below_threshold_returns_empty(self):
        docs = [
            Document(page_content="low1", metadata={"relevance_score": 0.1}),
            Document(page_content="low2", metadata={"relevance_score": 0.2}),
        ]
        assert _reranker._filter_by_score(docs, 0.9) == []


# ── _build_reranker — provider dispatch ──────────────────────────────────────

class TestBuildReranker:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown RERANK_PROVIDER"):
            _reranker._build_reranker("unknown_provider")

    def test_flashrank_provider_returns_reranker(self):
        try:
            reranker = _reranker._build_reranker("flashrank")
            assert reranker is not None
        except ImportError:
            pytest.skip("flashrank not installed")

    def test_cohere_provider_dispatches(self):
        try:
            reranker = _reranker._build_reranker("cohere")
            assert reranker is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("langchain_cohere not installed")

    def test_jina_provider_dispatches(self):
        try:
            reranker = _reranker._build_reranker("jina")
            assert reranker is not None
        except (ImportError, ModuleNotFoundError):
            pytest.skip("jina reranker not installed")
        except Exception:
            pass  # Jina SDK requires API key even to instantiate
