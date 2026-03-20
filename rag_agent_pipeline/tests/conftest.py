"""Shared fixtures and environment setup for all tests.

Two modes:
  pytest tests/               → default: LLM is mocked, no API needed
  pytest tests/ --llm         → integration: real LLM API required
"""

import os
import tempfile
import time
import uuid
from unittest.mock import MagicMock

# Configure services BEFORE importing any pipeline modules.
os.environ.setdefault("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL", "gemini-2.5-flash")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

import pytest
import fitz
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


# ── CLI option ────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--llm", action="store_true", default=False,
        help="Run tests against a real LLM API. Without this flag, LLM calls are mocked.",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "llm: test requires a real LLM")


def pytest_collection_modifyitems(config, items):
    """In default mode, auto-mock LLM for @pytest.mark.llm tests instead of skipping."""
    # Nothing to skip — we handle mocking in the fixture below.
    pass


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
def wait_for_services(request):
    """Block until Qdrant is reachable. LLM API is only checked when --llm is set."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")

    for _ in range(60):
        if _url_ready(f"{qdrant_url}/healthz"):
            break
        time.sleep(1)
    else:
        pytest.fail(f"Qdrant not ready after 60 s ({qdrant_url})")

    if request.config.getoption("--llm"):
        llm_base = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
        # Pour les API cloud (Gemini, OpenAI), on vérifie simplement que l'URL est accessible
        if _url_ready(llm_base):
            pass  # API accessible
        else:
            pytest.fail(f"LLM API not reachable ({llm_base})")


@pytest.fixture(scope="session", autouse=True)
def warmup_embeddings(wait_for_services):
    """Download / warm-up FastEmbed models once per session."""
    from config.embeddings import EMBEDDINGS, SPARSE_EMBEDDINGS
    EMBEDDINGS.embed_query("warmup")
    SPARSE_EMBEDDINGS.embed_query("warmup")


# ── LLM mock (default mode) ──────────────────────────────────────────────────

def _make_fake_llm():
    """Return a MagicMock that behaves like ChatOpenAI for basic invoke/stream."""
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content='{"page": 5}')

    def _stream(prompt, **kw):
        yield MagicMock(content="Mocked answer from LLM.")
    fake.stream.side_effect = _stream
    return fake


@pytest.fixture(autouse=True)
def _auto_mock_llm(request, monkeypatch):
    """When --llm is NOT passed, replace LLM with a mock in every module that uses it.

    Tests marked @pytest.mark.llm get the mock automatically in default mode.
    Tests NOT marked llm never touch the LLM so the mock is harmless.
    """
    if request.config.getoption("--llm"):
        return  # real LLM mode — do nothing

    fake = _make_fake_llm()

    import sys

    # Patch LLM everywhere it is imported.
    # Each node imports its dedicated LLM as "LLM" (via `import LLM_X as LLM`).
    # We also patch the per-step instances in config.llm for any direct access.
    module_keys = [
        "config.llm",
        "nodes.generator",
        "nodes.hyde",
        "nodes.query_rewriter",
        "nodes.query_expander",
        "nodes.self_query",
        "api.routes.chat",
    ]
    for key in module_keys:
        mod = sys.modules.get(key)
        if not mod:
            continue
        # Patche l'alias LLM dans chaque nœud
        if hasattr(mod, "LLM"):
            monkeypatch.setattr(mod, "LLM", fake)
        # Patche les instances per-step dans config.llm
        for attr in ("LLM_REWRITE", "LLM_EXPAND", "LLM_HYDE", "LLM_SELF_QUERY", "LLM_GENERATE"):
            if hasattr(mod, attr):
                monkeypatch.setattr(mod, attr, fake)


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
