"""Shared fixtures for API route tests."""

import time

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.state import sessions, jobs


@pytest.fixture(autouse=True)
def clean_state():
    """Clear sessions and jobs before each test."""
    sessions.clear()
    jobs.clear()
    yield
    sessions.clear()
    jobs.clear()


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def ingested_session(client, sample_pdf_bytes, test_collection):
    """Ingest a real PDF through the API and return (session_id, job_id)."""
    r = client.post(
        "/api/ingest",
        files={"file": ("test.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 202
    data = r.json()
    time.sleep(1)  # allow Qdrant indexing
    return data["session_id"], data["job_id"]
