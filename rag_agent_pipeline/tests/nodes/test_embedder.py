"""Tests for nodes.embedder — vector embedding and storage."""

from unittest.mock import MagicMock, patch


class TestEmbedAndStore:
    @patch("nodes.embedder.QdrantVectorStore.from_documents")
    def test_returns_ingested(self, mock_from_docs, base_state, sample_chunks):
        from nodes.embedder import embed_and_store

        mock_from_docs.return_value = MagicMock()
        result = embed_and_store(base_state(chunks=sample_chunks))
        assert result == {"ingested": True}

    @patch("nodes.embedder.QdrantVectorStore.from_documents")
    def test_calls_from_documents_with_sparse(self, mock_from_docs, base_state, sample_chunks):
        from nodes.embedder import embed_and_store

        mock_from_docs.return_value = MagicMock()
        embed_and_store(base_state(chunks=sample_chunks))

        mock_from_docs.assert_called_once()
        args = mock_from_docs.call_args
        assert len(args.kwargs["documents"]) == len(sample_chunks)
        assert "sparse_embedding" in args.kwargs

    def test_empty_chunks_skips_embedding(self, base_state):
        from nodes.embedder import embed_and_store

        result = embed_and_store(base_state(chunks=[]))
        assert result == {"ingested": False}
