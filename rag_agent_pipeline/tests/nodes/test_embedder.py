"""Tests for nodes.embedder — vector embedding and storage (real Qdrant)."""

import time


class TestEmbedAndStore:
    def test_returns_ingested(self, base_state, sample_chunks, test_collection):
        from nodes.embedder import embed_and_store

        result = embed_and_store(base_state(chunks=sample_chunks))
        assert result == {"ingested": True}

    def test_stores_documents_in_qdrant(self, base_state, sample_chunks, test_collection):
        from nodes.embedder import embed_and_store
        from config.qdrant import get_client

        embed_and_store(base_state(chunks=sample_chunks))
        time.sleep(1)

        info = get_client().get_collection(test_collection)
        assert info.points_count == len(sample_chunks)

    def test_empty_chunks_skips_embedding(self, base_state):
        from nodes.embedder import embed_and_store

        result = embed_and_store(base_state(chunks=[]))
        assert result == {"ingested": False}
