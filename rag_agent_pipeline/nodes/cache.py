"""Node — semantic cache: skip pipeline for repeated/similar questions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage

from config.pipeline import (
    CACHE_ENABLED,
    CACHE_MAX_SIZE,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL,
)
from config.embeddings import EMBEDDINGS
from logger import get_logger
from state import RAGState

log = get_logger("nodes.cache")


@dataclass
class CacheEntry:
    question_embedding: list[float]
    answer: str
    source_pages: list[int]
    timestamp: float


class SemanticCache:
    """In-memory semantic cache with cosine similarity matching."""

    def __init__(self) -> None:
        self._entries: list[CacheEntry] = []

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def lookup(self, question_embedding: list[float]) -> CacheEntry | None:
        """Find a cached entry with similarity above threshold."""
        now = time.time()
        best_entry = None
        best_score = 0.0

        for entry in self._entries:
            if now - entry.timestamp > CACHE_TTL:
                continue
            score = self._cosine_similarity(question_embedding, entry.question_embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= CACHE_SIMILARITY_THRESHOLD:
            log.info("Cache hit (similarity=%.3f)", best_score)
            return best_entry

        return None

    def store(self, question_embedding: list[float], answer: str, source_pages: list[int]) -> None:
        """Store a new cache entry, evicting oldest if at capacity."""
        now = time.time()

        # Evict expired entries
        self._entries = [e for e in self._entries if now - e.timestamp <= CACHE_TTL]

        # Evict oldest if at capacity
        if len(self._entries) >= CACHE_MAX_SIZE:
            self._entries.sort(key=lambda e: e.timestamp)
            self._entries = self._entries[-(CACHE_MAX_SIZE - 1):]

        self._entries.append(CacheEntry(
            question_embedding=question_embedding,
            answer=answer,
            source_pages=source_pages,
            timestamp=now,
        ))
        log.debug("Cached answer (%d entries total)", len(self._entries))

    def clear(self) -> None:
        count = len(self._entries)
        self._entries.clear()
        log.info("Cache cleared (%d entries removed)", count)

    @property
    def size(self) -> int:
        return len(self._entries)


# Global singleton — persists across requests within the same process
_cache = SemanticCache()


def get_cache() -> SemanticCache:
    """Return the global cache instance."""
    return _cache


def cache_check(state: RAGState) -> dict:
    """
    Check if a similar question was already answered.

    - Embeds the question and searches the cache by cosine similarity
    - If hit: returns cached answer, sets cache_hit=True (skips rest of pipeline)
    - If miss: sets cache_hit=False (pipeline continues normally)
    - Skipped when CACHE_ENABLED=false
    """
    if not CACHE_ENABLED:
        return {"cache_hit": False}

    question = state["question"]

    try:
        embedding = EMBEDDINGS.embed_query(question)
        entry = _cache.lookup(embedding)

        if entry:
            return {
                "cache_hit": True,
                "answer": entry.answer,
                "context": [],
            }

    except Exception as e:
        log.warning("Cache check failed: %s", e)

    return {"cache_hit": False}


def cache_store(state: RAGState) -> dict:
    """
    Store the answer in the semantic cache after generation.

    - Only stores if CACHE_ENABLED=true and answer was not from cache
    - Embeds the original question for future lookups
    """
    if not CACHE_ENABLED:
        return {}

    if state.get("cache_hit", False):
        return {}

    question = state.get("original_question") or state["question"]
    answer = state.get("answer", "")

    if not answer:
        return {}

    try:
        embedding = EMBEDDINGS.embed_query(question)
        source_pages = sorted(set(
            d.metadata.get("page", 0)
            for d in state.get("context", [])
            if d.metadata.get("page")
        ))
        _cache.store(embedding, answer, source_pages)
    except Exception as e:
        log.warning("Cache store failed: %s", e)

    return {}
