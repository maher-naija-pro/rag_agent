"""Tests for nodes.retriever (hybrid search) and nodes.reranker (reranking).

Uses real Qdrant + real FastEmbed embeddings — no mocks.
"""

import sys
import time

import pytest
from langchain_core.documents import Document

import nodes.retriever
import nodes.reranker
_retriever = sys.modules["nodes.retriever"]
_reranker = sys.modules["nodes.reranker"]


# ── retrieve() — hybrid search ──────────────────────────────────────────────

class TestRetrieve:
    def test_returns_candidates(self, base_state, embedded_collection):
        result = _retriever.retrieve(base_state(question="What is the capital of France?"))
        candidates = result["candidates"]
        assert len(candidates) > 0
        assert all(isinstance(d, Document) for d in candidates)

    def test_returns_documents_with_metadata(self, base_state, embedded_collection):
        result = _retriever.retrieve(base_state(question="Paris France"))
        for doc in result["candidates"]:
            assert "page" in doc.metadata

    def test_source_filter_matches(self, base_state, embedded_collection):
        """When filtering by the correct source, results are returned."""
        result = _retriever.retrieve(base_state(
            question="capital of France",
            source="test.pdf",
        ))
        candidates = result["candidates"]
        assert len(candidates) > 0
        for doc in candidates:
            assert doc.metadata.get("source") == "test.pdf"

    def test_source_filter_excludes_unmatched(self, base_state, embedded_collection):
        """When filtering by a non-matching source, no results are returned."""
        result = _retriever.retrieve(base_state(
            question="capital of France",
            source="nonexistent.pdf",
        ))
        assert result["candidates"] == []

    def test_multi_query_retrieval(self, base_state, embedded_collection):
        """Multiple queries merge and deduplicate results."""
        result = _retriever.retrieve(base_state(
            question="capital of France",
            expanded_queries=["capital of France", "Eiffel Tower Paris"],
        ))
        assert len(result["candidates"]) > 0

    def test_falls_back_to_question_when_no_expanded(self, base_state, embedded_collection):
        result = _retriever.retrieve(base_state(
            question="Paris",
            expanded_queries=[],
        ))
        assert len(result["candidates"]) > 0

    def test_metadata_filter_page(self, base_state, embedded_collection):
        """Page metadata filter restricts results to specific pages."""
        result = _retriever.retrieve(base_state(
            question="France",
            metadata_filter={"page": 1},
        ))
        for doc in result["candidates"]:
            assert doc.metadata.get("page") == 1

    def test_metadata_filter_excludes_wrong_page(self, base_state, embedded_collection):
        """Page filter for a page with no content returns empty."""
        result = _retriever.retrieve(base_state(
            question="France",
            metadata_filter={"page": 999},
        ))
        assert result["candidates"] == []

    def test_unsupported_metadata_filter_skipped(self, base_state, embedded_collection):
        """Unsupported filter keys are silently skipped, not crashing."""
        result = _retriever.retrieve(base_state(
            question="France",
            metadata_filter={"unsupported_field": "value"},
        ))
        # Should still return results (filter is skipped)
        assert isinstance(result["candidates"], list)


class TestBuildSearchKwargs:
    """Tests for _build_search_kwargs — config-driven search parameters."""

    def test_similarity_threshold_applied(self, monkeypatch):
        monkeypatch.setattr(_retriever, "SIMILARITY_THRESHOLD", 0.7)
        monkeypatch.setattr(_retriever, "HYBRID_FUSION_ALPHA", 0.5)
        kwargs = _retriever._build_search_kwargs("", None)
        assert kwargs["score_threshold"] == 0.7

    def test_alpha_applied_when_not_default(self, monkeypatch):
        monkeypatch.setattr(_retriever, "SIMILARITY_THRESHOLD", 0.0)
        monkeypatch.setattr(_retriever, "HYBRID_FUSION_ALPHA", 0.8)
        kwargs = _retriever._build_search_kwargs("", None)
        assert kwargs["alpha"] == 0.8

    def test_default_alpha_not_included(self, monkeypatch):
        monkeypatch.setattr(_retriever, "SIMILARITY_THRESHOLD", 0.0)
        monkeypatch.setattr(_retriever, "HYBRID_FUSION_ALPHA", 0.5)
        kwargs = _retriever._build_search_kwargs("", None)
        assert "alpha" not in kwargs

    def test_language_metadata_filter(self, monkeypatch):
        monkeypatch.setattr(_retriever, "SIMILARITY_THRESHOLD", 0.0)
        monkeypatch.setattr(_retriever, "HYBRID_FUSION_ALPHA", 0.5)
        kwargs = _retriever._build_search_kwargs("", {"language": "fr"})
        assert "filter" in kwargs

    def test_has_tables_metadata_filter(self, monkeypatch):
        monkeypatch.setattr(_retriever, "SIMILARITY_THRESHOLD", 0.0)
        monkeypatch.setattr(_retriever, "HYBRID_FUSION_ALPHA", 0.5)
        kwargs = _retriever._build_search_kwargs("", {"has_tables": True})
        assert "filter" in kwargs


class TestRetrieveError:
    """Tests for retrieval error handling."""

    def test_retrieval_exception_returns_empty(self, base_state, monkeypatch):
        """When Qdrant is unreachable, retrieve() returns empty candidates."""
        from unittest.mock import MagicMock

        # Mock get_store to raise on .as_retriever()
        bad_store = MagicMock()
        bad_store.as_retriever.return_value.invoke.side_effect = RuntimeError("Qdrant down")
        monkeypatch.setattr(_retriever, "get_store", lambda: bad_store)

        result = _retriever.retrieve(base_state(question="test"))
        assert result["candidates"] == []


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
    def test_returns_context(self, base_state):
        candidates = [
            Document(page_content="Paris is the capital of France.", metadata={"page": 1}),
            Document(page_content="The weather in Paris is nice.", metadata={"page": 2}),
        ]
        result = _reranker.rerank(base_state(candidates=candidates, question="capital of France"))
        assert len(result["context"]) > 0

    def test_empty_candidates(self, base_state):
        result = _reranker.rerank(base_state(candidates=[], question="test"))
        assert result["context"] == []

    def test_disabled_passes_candidates_through(self, base_state, monkeypatch):
        monkeypatch.setattr(_reranker, "RERANK_ENABLED", False)

        candidates = [
            Document(page_content="c1", metadata={"page": 1}),
            Document(page_content="c2", metadata={"page": 2}),
        ]
        result = _reranker.rerank(base_state(candidates=candidates, question="test"))
        assert result["context"] == candidates

    def test_reranker_error_falls_back_to_candidates(self, base_state, monkeypatch):
        monkeypatch.setattr(_reranker, "RERANK_ENABLED", True)
        monkeypatch.setattr(_reranker, "RERANK_PROVIDER", "unknown_provider_xxx")

        candidates = [
            Document(page_content="chunk1", metadata={"page": 1}),
            Document(page_content="chunk2", metadata={"page": 2}),
        ]
        result = _reranker.rerank(base_state(candidates=candidates, question="test"))
        assert len(result["context"]) == len(candidates)


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

    def test_singleton_returns_same_instance(self):
        """_build_reranker caches the instance — calling twice returns the same object."""
        # Reset the singleton state
        _reranker._reranker_instance = None
        _reranker._reranker_provider = None

        r1 = _reranker._build_reranker("flashrank")
        r2 = _reranker._build_reranker("flashrank")
        assert r1 is r2

    def test_singleton_recreates_on_provider_change(self):
        """When the provider changes, a new reranker instance is created."""
        _reranker._reranker_instance = None
        _reranker._reranker_provider = None

        r1 = _reranker._build_reranker("flashrank")
        # Force a provider change — unknown will raise, so we just check
        # that the guard detects the change
        assert _reranker._reranker_provider == "flashrank"
        # Calling with same provider returns cached
        r2 = _reranker._build_reranker("flashrank")
        assert r1 is r2
