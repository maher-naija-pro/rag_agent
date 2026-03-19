"""Node 3 — embed_and_store: embed chunks and upsert into Qdrant (hybrid)."""

from __future__ import annotations

from langchain_qdrant import QdrantVectorStore, RetrievalMode

from config import (
    COLLECTION,
    EMBEDDINGS,
    SPARSE_EMBEDDINGS,
    QDRANT_API_KEY,
    QDRANT_URL,
    set_store,
)
from logger import get_logger
from state import RAGState

log = get_logger("nodes.embedder")


def embed_and_store(state: RAGState) -> dict:
    """Embed chunks (dense + sparse) and upsert into Qdrant for hybrid search."""
    chunks = state["chunks"]

    if not chunks:
        log.warning("No chunks to embed")
        return {"ingested": False}

    log.info("Embedding %d chunks (dense + sparse) …", len(chunks))
    try:
        store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=EMBEDDINGS,
            sparse_embedding=SPARSE_EMBEDDINGS,
            collection_name=COLLECTION,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            retrieval_mode=RetrievalMode.HYBRID,
        )
    except Exception as e:
        log.error("Failed to embed/upsert to Qdrant: %s", e)
        raise

    set_store(store)
    log.info("Upserted to Qdrant at '%s' (hybrid)", QDRANT_URL)
    return {"ingested": True}
