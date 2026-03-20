"""Tests for configuration loading and defaults."""

import os


class TestPipelineConfig:
    def test_chunk_size_is_int(self):
        from config.pipeline import CHUNK_SIZE
        assert isinstance(CHUNK_SIZE, int) and CHUNK_SIZE > 0

    def test_chunk_overlap_less_than_size(self):
        from config.pipeline import CHUNK_SIZE, CHUNK_OVERLAP
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_retrieval_k_positive(self):
        from config.pipeline import RETRIEVAL_K
        assert RETRIEVAL_K > 0

    def test_rerank_top_n_lte_retrieval_k(self):
        from config.pipeline import RETRIEVAL_K, RERANK_TOP_N
        assert RERANK_TOP_N <= RETRIEVAL_K

    def test_similarity_threshold_in_range(self):
        from config.pipeline import SIMILARITY_THRESHOLD
        assert 0.0 <= SIMILARITY_THRESHOLD <= 1.0

    def test_hybrid_fusion_alpha_in_range(self):
        from config.pipeline import HYBRID_FUSION_ALPHA
        assert 0.0 <= HYBRID_FUSION_ALPHA <= 1.0

    def test_rerank_score_threshold_in_range(self):
        from config.pipeline import RERANK_SCORE_THRESHOLD
        assert 0.0 <= RERANK_SCORE_THRESHOLD <= 1.0

    def test_rerank_provider_is_valid(self):
        from config.pipeline import RERANK_PROVIDER
        assert RERANK_PROVIDER in ("flashrank", "cohere", "jina")

    def test_ocr_dpi_positive(self):
        from config.pipeline import OCR_DPI
        assert OCR_DPI > 0


class TestOptionalNodeFlags:
    """All optional nodes must have a boolean on/off flag."""

    def test_cache_enabled_is_bool(self):
        from config.pipeline import CACHE_ENABLED
        assert isinstance(CACHE_ENABLED, bool)

    def test_cache_ttl_positive(self):
        from config.pipeline import CACHE_TTL
        assert isinstance(CACHE_TTL, int) and CACHE_TTL > 0

    def test_cache_similarity_threshold_in_range(self):
        from config.pipeline import CACHE_SIMILARITY_THRESHOLD
        assert 0.0 <= CACHE_SIMILARITY_THRESHOLD <= 1.0

    def test_cache_max_size_positive(self):
        from config.pipeline import CACHE_MAX_SIZE
        assert isinstance(CACHE_MAX_SIZE, int) and CACHE_MAX_SIZE > 0

    def test_metadata_extraction_enabled_is_bool(self):
        from config.pipeline import METADATA_EXTRACTION_ENABLED
        assert isinstance(METADATA_EXTRACTION_ENABLED, bool)

    def test_query_rewrite_enabled_is_bool(self):
        from config.pipeline import QUERY_REWRITE_ENABLED
        assert isinstance(QUERY_REWRITE_ENABLED, bool)

    def test_query_expansion_enabled_is_bool(self):
        from config.pipeline import QUERY_EXPANSION_ENABLED
        assert isinstance(QUERY_EXPANSION_ENABLED, bool)

    def test_hyde_enabled_is_bool(self):
        from config.pipeline import HYDE_ENABLED
        assert isinstance(HYDE_ENABLED, bool)

    def test_self_query_enabled_is_bool(self):
        from config.pipeline import SELF_QUERY_ENABLED
        assert isinstance(SELF_QUERY_ENABLED, bool)

    def test_rerank_enabled_is_bool(self):
        from config.pipeline import RERANK_ENABLED
        assert isinstance(RERANK_ENABLED, bool)

    def test_query_expansion_count_positive(self):
        from config.pipeline import QUERY_EXPANSION_COUNT
        assert isinstance(QUERY_EXPANSION_COUNT, int) and QUERY_EXPANSION_COUNT > 0


class TestServerConfig:
    def test_max_upload_size_positive(self):
        from config.server import MAX_UPLOAD_SIZE_MB, MAX_FILE_SIZE
        assert MAX_UPLOAD_SIZE_MB > 0
        assert MAX_FILE_SIZE == MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def test_api_port_valid(self):
        from config.server import API_PORT
        assert 1 <= API_PORT <= 65535

    def test_cors_origins_is_list(self):
        from config.server import CORS_ORIGINS
        assert isinstance(CORS_ORIGINS, list) and len(CORS_ORIGINS) >= 1

    def test_log_level_valid(self):
        from config.server import LOG_LEVEL
        assert LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class TestLLMConfig:
    def test_temperature_in_range(self):
        from config.llm import LLM_TEMPERATURE
        assert 0.0 <= LLM_TEMPERATURE <= 2.0

    def test_per_step_models_are_set(self):
        from config.llm import (
            LLM_MODEL_REWRITE, LLM_MODEL_EXPAND, LLM_MODEL_HYDE,
            LLM_MODEL_SELF_QUERY, LLM_MODEL_GENERATE,
        )
        # Chaque modèle par étape doit être une chaîne non vide
        for model in (LLM_MODEL_REWRITE, LLM_MODEL_EXPAND, LLM_MODEL_HYDE,
                      LLM_MODEL_SELF_QUERY, LLM_MODEL_GENERATE):
            assert isinstance(model, str) and len(model) > 0
