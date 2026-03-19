"""Tests for nodes.cache — semantic cache check and store."""

import sys
import time
from unittest.mock import MagicMock, patch

import nodes.cache as cache_module


class TestSemanticCache:
    """Unit tests for the SemanticCache class — no mocks, pure logic."""

    def test_store_and_lookup(self):
        cache = cache_module.SemanticCache()
        embedding = [1.0, 0.0, 0.0]

        cache.store(embedding, "cached answer", [1, 2])

        entry = cache.lookup(embedding)
        assert entry is not None
        assert entry.answer == "cached answer"
        assert entry.source_pages == [1, 2]

    def test_lookup_miss_on_different_embedding(self):
        cache = cache_module.SemanticCache()
        cache.store([1.0, 0.0, 0.0], "answer", [1])

        entry = cache.lookup([0.0, 1.0, 0.0])  # orthogonal = similarity 0
        assert entry is None

    def test_lookup_miss_on_empty_cache(self):
        cache = cache_module.SemanticCache()
        assert cache.lookup([1.0, 0.0]) is None

    @patch.object(cache_module, "CACHE_TTL", 1)
    def test_expired_entries_not_returned(self):
        cache = cache_module.SemanticCache()
        embedding = [1.0, 0.0]

        cache.store(embedding, "old answer", [1])
        # Manually expire
        cache._entries[0].timestamp = time.time() - 10

        assert cache.lookup(embedding) is None

    @patch.object(cache_module, "CACHE_MAX_SIZE", 2)
    def test_evicts_oldest_at_capacity(self):
        cache = cache_module.SemanticCache()

        cache.store([1.0, 0.0], "first", [1])
        cache.store([0.0, 1.0], "second", [2])
        cache.store([0.5, 0.5], "third", [3])

        assert cache.size <= 2

    def test_clear(self):
        cache = cache_module.SemanticCache()
        cache.store([1.0], "answer", [1])
        cache.clear()
        assert cache.size == 0

    def test_cosine_similarity_identical(self):
        cache = cache_module.SemanticCache()
        assert cache._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_cosine_similarity_orthogonal(self):
        cache = cache_module.SemanticCache()
        assert cache._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_cosine_similarity_zero_vector(self):
        cache = cache_module.SemanticCache()
        assert cache._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestCacheCheck:
    """Tests for cache_check() node function.

    Mocks: EMBEDDINGS (external embedding API).
    Never mocks: cache logic (our code under test).
    """

    @patch.object(cache_module, "CACHE_ENABLED", False)
    def test_disabled_returns_miss(self, base_state):
        result = cache_module.cache_check(base_state(question="test"))
        assert result["cache_hit"] is False

    @patch.object(cache_module, "CACHE_ENABLED", True)
    @patch.object(cache_module, "EMBEDDINGS")
    def test_miss_on_empty_cache(self, mock_embeddings, base_state):
        cache_module._cache.clear()
        mock_embeddings.embed_query.return_value = [1.0, 0.0, 0.0]

        result = cache_module.cache_check(base_state(question="new question"))
        assert result["cache_hit"] is False

    @patch.object(cache_module, "CACHE_ENABLED", True)
    @patch.object(cache_module, "EMBEDDINGS")
    def test_hit_returns_cached_answer(self, mock_embeddings, base_state):
        cache_module._cache.clear()
        embedding = [1.0, 0.0, 0.0]
        mock_embeddings.embed_query.return_value = embedding

        # Pre-populate cache
        cache_module._cache.store(embedding, "cached answer", [1, 3])

        result = cache_module.cache_check(base_state(question="same question"))

        assert result["cache_hit"] is True
        assert result["answer"] == "cached answer"

    @patch.object(cache_module, "CACHE_ENABLED", True)
    @patch.object(cache_module, "EMBEDDINGS")
    def test_fallback_on_embedding_error(self, mock_embeddings, base_state):
        mock_embeddings.embed_query.side_effect = RuntimeError("API down")

        result = cache_module.cache_check(base_state(question="test"))
        assert result["cache_hit"] is False


class TestCacheStore:
    """Tests for cache_store() node function."""

    @patch.object(cache_module, "CACHE_ENABLED", False)
    def test_disabled_does_nothing(self, base_state):
        result = cache_module.cache_store(base_state(question="test", answer="answer"))
        assert result == {}

    @patch.object(cache_module, "CACHE_ENABLED", True)
    def test_skips_on_cache_hit(self, base_state):
        result = cache_module.cache_store(base_state(
            question="test", answer="answer", cache_hit=True,
        ))
        assert result == {}

    @patch.object(cache_module, "CACHE_ENABLED", True)
    @patch.object(cache_module, "EMBEDDINGS")
    def test_stores_answer(self, mock_embeddings, base_state):
        cache_module._cache.clear()
        embedding = [0.5, 0.5, 0.0]
        mock_embeddings.embed_query.return_value = embedding

        cache_module.cache_store(base_state(
            question="my question",
            answer="my answer",
            cache_hit=False,
        ))

        assert cache_module._cache.size == 1

    @patch.object(cache_module, "CACHE_ENABLED", True)
    def test_skips_empty_answer(self, base_state):
        before = cache_module._cache.size
        cache_module.cache_store(base_state(question="test", answer="", cache_hit=False))
        assert cache_module._cache.size == before

    @patch.object(cache_module, "CACHE_ENABLED", True)
    @patch.object(cache_module, "EMBEDDINGS")
    def test_fallback_on_embedding_error(self, mock_embeddings, base_state):
        mock_embeddings.embed_query.side_effect = RuntimeError("API down")

        result = cache_module.cache_store(base_state(
            question="test", answer="answer", cache_hit=False,
        ))
        assert result == {}


class TestGraphRouting:
    """Tests for the graph routing functions related to cache."""

    def test_after_cache_check_hit(self, base_state):
        from graph import after_cache_check
        assert after_cache_check(base_state(cache_hit=True)) == "generate"

    def test_after_cache_check_miss(self, base_state):
        from graph import after_cache_check
        assert after_cache_check(base_state(cache_hit=False)) == "rewrite_query"
