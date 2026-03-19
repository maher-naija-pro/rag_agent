"""Node — retrieve: hybrid search (dense + sparse) from Qdrant."""

from __future__ import annotations

from langchain_core.documents import Document

from config import (
    HYBRID_FUSION_ALPHA,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
    get_store,
)
from logger import get_logger
from state import RAGState

log = get_logger("nodes.retriever")


def _build_search_kwargs(source: str, metadata_filter: dict | None = None) -> dict:
    """Build search_kwargs for the Qdrant retriever."""
    search_kwargs: dict = {"k": RETRIEVAL_K}

    if SIMILARITY_THRESHOLD > 0.0:
        search_kwargs["score_threshold"] = SIMILARITY_THRESHOLD

    if HYBRID_FUSION_ALPHA != 0.5:
        search_kwargs["alpha"] = HYBRID_FUSION_ALPHA

    # Build Qdrant filter conditions
    conditions = []

    if source:
        from qdrant_client.models import FieldCondition, MatchValue
        conditions.append(
            FieldCondition(key="metadata.source", match=MatchValue(value=source))
        )
        log.debug("Filtering retrieval to source='%s'", source)

    # Apply metadata filters extracted by self-query node
    if metadata_filter:
        from qdrant_client.models import FieldCondition, MatchValue
        for key, value in metadata_filter.items():
            if key == "page" and isinstance(value, int):
                conditions.append(
                    FieldCondition(key="metadata.page", match=MatchValue(value=value))
                )
            elif key == "language" and isinstance(value, str):
                conditions.append(
                    FieldCondition(key="metadata.language", match=MatchValue(value=value))
                )
            elif key == "has_tables" and isinstance(value, bool):
                conditions.append(
                    FieldCondition(key="metadata.has_tables", match=MatchValue(value=value))
                )
            else:
                log.debug("Skipping unsupported filter: %s=%s", key, value)
        log.info("Applied %d metadata filters: %s", len(metadata_filter), metadata_filter)

    if conditions:
        from qdrant_client.models import Filter
        search_kwargs["filter"] = Filter(must=conditions)

    return search_kwargs


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate documents by page_content."""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        key = doc.page_content
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def retrieve(state: RAGState) -> dict:
    """
    Hybrid search (dense + BM25 sparse) → candidates from Qdrant.

    When expanded_queries is populated (query expansion enabled),
    runs retrieval for each variant and merges + deduplicates results.
    Otherwise uses the single question.
    """
    queries = state.get("expanded_queries", [])
    if not queries:
        queries = [state["question"]]

    source = state.get("source", "")
    metadata_filter = state.get("metadata_filter", {})
    search_kwargs = _build_search_kwargs(source, metadata_filter)

    try:
        base_retriever = get_store().as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        all_candidates: list[Document] = []
        for query in queries:
            results = base_retriever.invoke(query)
            all_candidates.extend(results)

        candidates = _deduplicate(all_candidates)

    except Exception as e:
        log.error("Retrieval failed: %s", e)
        return {"candidates": []}

    pages_hit = [d.metadata.get("page", "?") for d in candidates]
    log.info(
        "Retrieved %d candidates (%d queries, %d before dedup) %s",
        len(candidates), len(queries), len(all_candidates), pages_hit,
    )
    return {"candidates": candidates}
