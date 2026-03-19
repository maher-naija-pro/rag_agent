"""Shared fixtures and environment setup for all tests.

All external services (Qdrant, Ollama) run via docker-compose.test.yml.
No unittest.mock — tests hit real services.
"""

import os
import tempfile
import time
import uuid

# Configure services BEFORE importing any pipeline modules.
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("LLM_MODEL", "mistral:7b")
os.environ.setdefault("OPENAI_API_KEY", "ollama")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

import pytest
import fitz
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


# ── Service readiness ─────────────────────────────────────────────────────────

def _url_ready(url: str, timeout: float = 2) -> bool:
    from urllib.request import urlopen
    from urllib.error import URLError
    try:
        urlopen(url, timeout=timeout)
        return True
    except (URLError, OSError):
        return False


@pytest.fixture(scope="session", autouse=True)
def wait_for_services():
    """Block until Qdrant and Ollama are reachable."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    ollama_base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    # Derive Ollama root URL (strip /v1 suffix) for the tags endpoint
    ollama_root = ollama_base.replace("/v1", "")

    for _ in range(60):
        if _url_ready(f"{qdrant_url}/healthz"):
            break
        time.sleep(1)
    else:
        pytest.fail(f"Qdrant not ready after 60 s ({qdrant_url})")

    for _ in range(60):
        if _url_ready(f"{ollama_root}/api/tags"):
            break
        time.sleep(1)
    else:
        pytest.fail(f"Ollama not ready after 60 s ({ollama_root})")


@pytest.fixture(scope="session", autouse=True)
def warmup_embeddings(wait_for_services):
    """Download / warm-up FastEmbed models once per session."""
    from config.embeddings import EMBEDDINGS, SPARSE_EMBEDDINGS
    EMBEDDINGS.embed_query("warmup")
    SPARSE_EMBEDDINGS.embed_query("warmup")


# ── Qdrant isolation ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_qdrant_store():
    """Reset the Qdrant store singleton between tests."""
    import config.qdrant as qdrant_mod
    qdrant_mod._store = None
    yield
    qdrant_mod._store = None


@pytest.fixture()
def test_collection(monkeypatch):
    """Provide a unique Qdrant collection; delete it after the test."""
    import config.qdrant as qdrant_mod
    import config as config_mod
    import nodes.embedder as embedder_mod

    name = f"test_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(qdrant_mod, "COLLECTION", name)
    monkeypatch.setattr(config_mod, "COLLECTION", name)
    monkeypatch.setattr(embedder_mod, "COLLECTION", name)
    monkeypatch.setattr(qdrant_mod, "_store", None)

    yield name

    try:
        qdrant_mod.get_client().delete_collection(name)
    except Exception:
        pass


@pytest.fixture()
def embedded_collection(test_collection, sample_chunks):
    """Embed sample_chunks into a unique Qdrant collection."""
    from nodes.embedder import embed_and_store

    state = {
        "messages": [],
        "question": "",
        "original_question": "",
        "expanded_queries": [],
        "metadata_filter": {},
        "source": "",
        "raw_pages": [],
        "chunks": sample_chunks,
        "candidates": [],
        "context": [],
        "answer": "",
        "cache_hit": False,
        "ingested": False,
    }
    embed_and_store(state)
    time.sleep(1)  # allow Qdrant to index
    return test_collection


# ── Reusable state / data fixtures ────────────────────────────────────────────

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
