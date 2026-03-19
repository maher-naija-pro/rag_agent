"""Pipeline tuning parameters — OCR, chunking, retrieval, reranking."""

import os

import config.env  # noqa: F401  ensure .env is loaded

# OCR
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "fra+eng")
OCR_DPI       = int(os.getenv("OCR_DPI", "300"))

# Chunking
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Retrieval
RETRIEVAL_K          = int(os.getenv("RETRIEVAL_K", "15"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
HYBRID_FUSION_ALPHA  = float(os.getenv("HYBRID_FUSION_ALPHA", "0.5"))

# Semantic cache
CACHE_ENABLED              = os.getenv("CACHE_ENABLED", "false").lower() in ("true", "1", "yes")
CACHE_TTL                  = int(os.getenv("CACHE_TTL", "3600"))
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
CACHE_MAX_SIZE             = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Query rewriting
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() in ("true", "1", "yes")

# Query expansion
QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() in ("true", "1", "yes")
QUERY_EXPANSION_COUNT   = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))

# Self-query (metadata filter extraction)
SELF_QUERY_ENABLED = os.getenv("SELF_QUERY_ENABLED", "true").lower() in ("true", "1", "yes")

# HyDE (Hypothetical Document Embedding)
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "false").lower() in ("true", "1", "yes")

# Metadata extraction
METADATA_EXTRACTION_ENABLED = os.getenv("METADATA_EXTRACTION_ENABLED", "true").lower() in ("true", "1", "yes")
METADATA_FIELDS = [f.strip() for f in os.getenv("METADATA_FIELDS", "dates,keywords,language").split(",") if f.strip()]

# Reranking
RERANK_ENABLED         = os.getenv("RERANK_ENABLED", "true").lower() in ("true", "1", "yes")
RERANK_PROVIDER        = os.getenv("RERANK_PROVIDER", "flashrank")
RERANK_MODEL           = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
RERANK_TOP_N           = int(os.getenv("RERANK_TOP_N", "4"))
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.1"))
RERANK_API_KEY         = os.getenv("RERANK_API_KEY", "")
