"""Tests for nodes.cache — semantic cache check and store (real FastEmbed)."""

import sys
import time

import nodes.cache as cache_module


class TestSemanticCache:
    """Unit tests for the SemanticCache class — pure logic, no external services."""

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

    def test_expired_entries_not_returned(self, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_TTL", 1)

        cache = cache_module.SemanticCache()
        embedding = [1.0, 0.0]

        cache.store(embedding, "old answer", [1])
        # Manually expire
        cache._entries[0].timestamp = time.time() - 10

        assert cache.lookup(embedding) is None

    def test_evicts_oldest_at_capacity(self, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_MAX_SIZE", 2)

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
    """Tests for cache_check() node function using real FastEmbed embeddings."""

    def test_disabled_returns_miss(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", False)

        result = cache_module.cache_check(base_state(question="test"))
        assert result["cache_hit"] is False

    def test_miss_on_empty_cache(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", True)
        cache_module._cache.clear()

        result = cache_module.cache_check(base_state(question="new question"))
        assert result["cache_hit"] is False

    def test_hit_returns_cached_answer(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", True)
        cache_module._cache.clear()

        # Use real embeddings to embed and store
        from config.embeddings import EMBEDDINGS
        embedding = EMBEDDINGS.embed_query("same question")
        cache_module._cache.store(embedding, "cached answer", [1, 3])

        result = cache_module.cache_check(base_state(question="same question"))

        assert result["cache_hit"] is True
        assert result["answer"] == "cached answer"

    def test_fallback_on_embedding_error(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", True)

        # Use a broken embeddings object
        class BrokenEmbeddings:
            def embed_query(self, text):
                raise RuntimeError("API down")

        monkeypatch.setattr(cache_module, "EMBEDDINGS", BrokenEmbeddings())

        result = cache_module.cache_check(base_state(question="test"))
        assert result["cache_hit"] is False


class TestCacheStore:
    """Tests for cache_store() node function."""

    def test_disabled_does_nothing(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", False)

        result = cache_module.cache_store(base_state(question="test", answer="answer"))
        assert result == {}

    def test_skips_on_cache_hit(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", True)

        result = cache_module.cache_store(base_state(
            question="test", answer="answer", cache_hit=True,
        ))
        assert result == {}

    def test_stores_answer(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", True)
        cache_module._cache.clear()

        cache_module.cache_store(base_state(
            question="my question",
            answer="my answer",
            cache_hit=False,
        ))

        assert cache_module._cache.size == 1

    def test_skips_empty_answer(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", True)

        before = cache_module._cache.size
        cache_module.cache_store(base_state(question="test", answer="", cache_hit=False))
        assert cache_module._cache.size == before

    def test_fallback_on_embedding_error(self, base_state, monkeypatch):
        monkeypatch.setattr(cache_module, "CACHE_ENABLED", True)

        class BrokenEmbeddings:
            def embed_query(self, text):
                raise RuntimeError("API down")

        monkeypatch.setattr(cache_module, "EMBEDDINGS", BrokenEmbeddings())

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
