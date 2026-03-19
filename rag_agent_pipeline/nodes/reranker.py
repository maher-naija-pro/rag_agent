"""Node 5 — rerank: re-score candidates and keep the best N."""

from __future__ import annotations

from langchain_core.documents import Document

from config import (
    RERANK_API_KEY,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_PROVIDER,
    RERANK_SCORE_THRESHOLD,
    RERANK_TOP_N,
)
from logger import get_logger
from state import RAGState

log = get_logger("nodes.reranker")


def _build_reranker(provider: str | None = None):
    """Build a reranker based on RERANK_PROVIDER."""
    if provider is None:
        provider = RERANK_PROVIDER
    provider = provider.lower().strip()

    if provider == "flashrank":
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        return FlashrankRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)

    if provider == "cohere":
        from langchain_cohere import CohereRerank
        return CohereRerank(
            model=RERANK_MODEL or "rerank-v3.5",
            top_n=RERANK_TOP_N,
            cohere_api_key=RERANK_API_KEY,
        )

    if provider == "jina":
        from langchain_community.document_compressors.jina_rerank import JinaRerank
        return JinaRerank(
            model=RERANK_MODEL or "jina-reranker-v2-base-multilingual",
            top_n=RERANK_TOP_N,
            jina_api_key=RERANK_API_KEY,
        )

    raise ValueError(
        f"Unknown RERANK_PROVIDER: '{provider}'. "
        "Supported: flashrank, cohere, jina"
    )


def _filter_by_score(docs: list[Document], threshold: float) -> list[Document]:
    """Drop documents below the rerank score threshold."""
    if threshold <= 0.0:
        return docs
    return [
        d for d in docs
        if d.metadata.get("relevance_score", 1.0) >= threshold
    ]


def rerank(state: RAGState) -> dict:
    """
    Stage 2 — Rerank candidates and keep the top-N most relevant.

    - RERANK_PROVIDER: flashrank, cohere, or jina
    - RERANK_MODEL: model name for the chosen provider
    - RERANK_TOP_N: how many to keep
    - RERANK_SCORE_THRESHOLD: minimum confidence to keep
    """
    candidates = state.get("candidates", [])
    question = state["question"]

    if not candidates:
        log.warning("No candidates to rerank")
        return {"context": []}

    if not RERANK_ENABLED:
        log.debug("Reranking disabled — passing candidates through as context")
        return {"context": candidates}

    try:
        reranker = _build_reranker(RERANK_PROVIDER)
        docs = reranker.compress_documents(candidates, question)
        docs = list(docs)
    except Exception as e:
        log.error("Reranking failed (provider=%s): %s", RERANK_PROVIDER, e)
        # Fall back to unranked candidates so the pipeline can still answer
        docs = candidates

    docs = _filter_by_score(docs, RERANK_SCORE_THRESHOLD)

    pages_hit = [d.metadata.get("page", "?") for d in docs]
    log.info("Reranked → %d docs %s (provider=%s)", len(docs), pages_hit, RERANK_PROVIDER)
    return {"context": docs}
