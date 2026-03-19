"""Tests for nodes.chunker — text splitting."""

from langchain_core.documents import Document


class TestChunk:
    def test_returns_chunks(self, base_state, sample_pages):
        from nodes.chunker import chunk

        result = chunk(base_state(raw_pages=sample_pages))
        assert len(result["chunks"]) >= 1

    def test_preserves_metadata(self, base_state):
        from nodes.chunker import chunk

        pages = [Document(page_content="Hello. " * 50, metadata={"page": 3, "source": "doc.pdf"})]
        chunks = chunk(base_state(raw_pages=pages))["chunks"]
        assert all(c.metadata.get("page") == 3 for c in chunks)

    def test_splits_long_document(self, base_state):
        from nodes.chunker import chunk

        pages = [Document(page_content="word " * 2000, metadata={"page": 1})]
        chunks = chunk(base_state(raw_pages=pages))["chunks"]
        assert len(chunks) > 1

    def test_empty_pages_produce_no_chunks(self, base_state):
        from nodes.chunker import chunk

        result = chunk(base_state(raw_pages=[]))
        assert result["chunks"] == []

    def test_short_text_stays_in_one_chunk(self, base_state):
        from nodes.chunker import chunk

        pages = [Document(page_content="Short text.", metadata={"page": 1})]
        chunks = chunk(base_state(raw_pages=pages))["chunks"]
        assert len(chunks) == 1
