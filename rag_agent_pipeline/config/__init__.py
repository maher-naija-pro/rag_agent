"""Configuration package — re-exports all settings for backward compatibility.

Modules:
    config.env        — .env loader (auto-imported)
    config.llm        — LLM singleton
    config.embeddings — Dense + sparse embedding models
    config.qdrant     — Qdrant client, store, collection
    config.pipeline   — Chunking, retrieval, reranking params
    config.server     — API host/port, CORS, upload, logging

Usage:
    from config import LLM, EMBEDDINGS, get_store
    from config.qdrant import get_client
    from config.pipeline import CHUNK_SIZE
"""

# Re-export everything so `from config import X` keeps working.

from config.llm import (  # noqa: F401
    LLM,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
)

from config.embeddings import (  # noqa: F401
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDINGS,
    SPARSE_EMBEDDINGS,
    SPARSE_MODEL,
)

from config.qdrant import (  # noqa: F401
    COLLECTION,
    QDRANT_API_KEY,
    QDRANT_URL,
    get_client,
    get_store,
    set_store,
)

from config.pipeline import (  # noqa: F401
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HYBRID_FUSION_ALPHA,
    METADATA_EXTRACTION_ENABLED,
    METADATA_FIELDS,
    CACHE_ENABLED,
    CACHE_MAX_SIZE,
    CACHE_SIMILARITY_THRESHOLD,
    CACHE_TTL,
    HYDE_ENABLED,
    SELF_QUERY_ENABLED,
    QUERY_EXPANSION_COUNT,
    QUERY_EXPANSION_ENABLED,
    QUERY_REWRITE_ENABLED,
    OCR_DPI,
    OCR_LANGUAGES,
    RERANK_ENABLED,
    RERANK_API_KEY,
    RERANK_MODEL,
    RERANK_PROVIDER,
    RERANK_SCORE_THRESHOLD,
    RERANK_TOP_N,
    RETRIEVAL_K,
    SIMILARITY_THRESHOLD,
)

from config.server import (  # noqa: F401
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    LOG_LEVEL,
    MAX_FILE_SIZE,
    MAX_UPLOAD_SIZE_MB,
    UPLOAD_DIR,
)
