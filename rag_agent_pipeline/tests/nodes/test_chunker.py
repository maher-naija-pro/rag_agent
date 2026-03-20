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

    def test_chunking_exception_returns_empty(self, base_state, monkeypatch):
        """When the splitter raises an error, chunk() returns an empty list gracefully."""
        from nodes.chunker import chunk
        import nodes.chunker as _mod
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Patch split_documents to raise an exception
        original_init = RecursiveCharacterTextSplitter.__init__

        def bad_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.split_documents = lambda docs: (_ for _ in ()).throw(RuntimeError("split failed"))

        monkeypatch.setattr(RecursiveCharacterTextSplitter, "__init__", bad_init)

        pages = [Document(page_content="Some text content.", metadata={"page": 1})]
        result = chunk(base_state(raw_pages=pages))
        assert result["chunks"] == []
