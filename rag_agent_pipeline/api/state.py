"""Shared in-memory state for sessions, jobs, and the LangGraph instance.

In production replace with Redis (sessions/jobs) and PostgresSaver (checkpointer).
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from config import UPLOAD_DIR, MAX_FILE_SIZE  # noqa: F401  re-exported for routes
from graph import build_graph

checkpointer = InMemorySaver()
graph = build_graph(checkpointer)

# session_id → session metadata
sessions: dict[str, dict[str, Any]] = {}

# job_id → job status
jobs: dict[str, dict[str, Any]] = {}

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
