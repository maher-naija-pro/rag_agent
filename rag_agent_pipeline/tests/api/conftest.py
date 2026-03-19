"""Shared fixtures for API route tests."""

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
