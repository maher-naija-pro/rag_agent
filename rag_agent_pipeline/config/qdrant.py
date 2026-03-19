"""Qdrant vector database — client, store, and collection settings."""

from __future__ import annotations

import os

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from config.embeddings import EMBEDDINGS, SPARSE_EMBEDDINGS

import config.env  # noqa: F401  ensure .env is loaded

COLLECTION     = os.getenv("QDRANT_COLLECTION", "pdf_rag")
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

_client: QdrantClient | None = None
_store: QdrantVectorStore | None = None


def get_client() -> QdrantClient:
    """Lazy singleton for the Qdrant client."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def get_store() -> QdrantVectorStore:
    """Lazy singleton for the hybrid vector store."""
    global _store
    if _store is None:
        _store = QdrantVectorStore(
            client=get_client(),
            collection_name=COLLECTION,
            embedding=EMBEDDINGS,
            sparse_embedding=SPARSE_EMBEDDINGS,
            retrieval_mode=RetrievalMode.HYBRID,
        )
    return _store


def set_store(store: QdrantVectorStore) -> None:
    """Replace the global store (called after ingestion)."""
    global _store
    _store = store
