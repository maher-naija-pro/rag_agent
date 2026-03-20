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

    def test_chunks_have_line_start_metadata(self, base_state):
        """Every chunk must have a line_start metadata field (1-indexed)."""
        from nodes.chunker import chunk

        pages = [Document(
            page_content="Line one\nLine two\nLine three\nLine four\nLine five",
            metadata={"page": 1, "source": "test.pdf"},
        )]
        chunks = chunk(base_state(raw_pages=pages))["chunks"]
        assert len(chunks) >= 1
        for c in chunks:
            assert "line_start" in c.metadata, f"Missing line_start in {c.metadata}"
            assert isinstance(c.metadata["line_start"], int)
            assert c.metadata["line_start"] >= 1

    def test_chunks_have_line_end_metadata(self, base_state):
        """Every chunk must have a line_end metadata field >= line_start."""
        from nodes.chunker import chunk

        pages = [Document(
            page_content="First line\nSecond line\nThird line",
            metadata={"page": 1, "source": "test.pdf"},
        )]
        chunks = chunk(base_state(raw_pages=pages))["chunks"]
        for c in chunks:
            assert "line_end" in c.metadata, f"Missing line_end in {c.metadata}"
            assert c.metadata["line_end"] >= c.metadata["line_start"]

    def test_line_numbers_are_sequential_across_chunks(self, base_state):
        """Chunks from the same page should have increasing line_start values."""
        from nodes.chunker import chunk

        text = "\n".join(f"This is line number {i} with enough text." for i in range(1, 51))
        pages = [Document(page_content=text, metadata={"page": 1, "source": "test.pdf"})]
        chunks = chunk(base_state(raw_pages=pages))["chunks"]

        if len(chunks) > 1:
            line_starts = [c.metadata["line_start"] for c in chunks]
            # line_start should be non-decreasing (overlap may cause equal values)
            for i in range(1, len(line_starts)):
                assert line_starts[i] >= line_starts[i - 1], (
                    f"line_start went backwards: {line_starts[i-1]} -> {line_starts[i]}"
                )

    def test_single_line_chunk_has_equal_start_end(self, base_state):
        """A chunk with no newlines should have line_start == line_end."""
        from nodes.chunker import chunk

        pages = [Document(page_content="Just one line here.", metadata={"page": 1, "source": "test.pdf"})]
        chunks = chunk(base_state(raw_pages=pages))["chunks"]
        assert len(chunks) == 1
        assert chunks[0].metadata["line_start"] == chunks[0].metadata["line_end"]

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
