"""Shared fixtures and environment setup for all tests."""

import os
import tempfile

# Set fake keys before any pipeline module is imported
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

import pytest
import fitz
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


@pytest.fixture()
def base_state():
    """Return a factory that builds a minimal RAGState dict."""
    def _make(**overrides) -> dict:
        state = {
            "messages":          [],
            "question":          "What is in this document?",
            "original_question": "",
            "expanded_queries":  [],
            "metadata_filter":   {},
            "source":            "",
            "raw_pages":         [],
            "chunks":            [],
            "candidates":        [],
            "context":           [],
            "answer":            "",
            "cache_hit":         False,
            "ingested":          False,
        }
        state.update(overrides)
        return state
    return _make


@pytest.fixture()
def sample_pdf():
    """Create a one-page PDF with text and return its path."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Hello from PyMuPDF test fixture")
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    doc.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture()
def sample_pdf_bytes():
    """Return raw PDF bytes for upload tests."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Test PDF content for API")
    buf = doc.tobytes()
    doc.close()
    return buf


@pytest.fixture()
def sample_pages():
    """Return a list of Document pages for chunker/embedder tests."""
    return [
        Document(page_content="This is a sentence. " * 100, metadata={"page": 1, "source": "test.pdf", "method": "native"}),
        Document(page_content="Another page content. " * 80, metadata={"page": 2, "source": "test.pdf", "method": "native"}),
    ]


@pytest.fixture()
def sample_chunks():
    """Return a list of small Document chunks for embedder/retriever tests."""
    return [
        Document(page_content="Paris is the capital of France.", metadata={"page": 1, "source": "test.pdf"}),
        Document(page_content="The Eiffel Tower is in Paris.", metadata={"page": 1, "source": "test.pdf"}),
        Document(page_content="France is in Europe.", metadata={"page": 2, "source": "test.pdf"}),
    ]


@pytest.fixture()
def sample_context():
    """Return reranked context documents for generator tests."""
    return [
        Document(page_content="Paris is the capital of France.", metadata={"page": 1}),
        Document(page_content="France is a country in Europe.", metadata={"page": 3}),
    ]
