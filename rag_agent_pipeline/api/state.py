"""Shared in-memory state for sessions, jobs, and the LangGraph instance.

In production replace with Redis (sessions/jobs) and PostgresSaver (checkpointer).
"""

# Active les annotations différées pour le typage
from __future__ import annotations

# Importe le type générique Any
from typing import Any

# Sauvegarde des checkpoints en mémoire
from langgraph.checkpoint.memory import InMemorySaver

# Importe le répertoire d'upload et la taille maximale de fichier
from config import UPLOAD_DIR, MAX_FILE_SIZE  # noqa: F401  re-exported for routes
# Fonction de construction du graphe LangGraph
from graph import build_graph

# Crée un checkpointer en mémoire pour sauvegarder l'état des conversations
checkpointer = InMemorySaver()
# Construit le graphe d'exécution du pipeline RAG
graph = build_graph(checkpointer)

# session_id → session metadata
# Dictionnaire associant chaque session à ses métadonnées
sessions: dict[str, dict[str, Any]] = {}

# job_id → job status
# Dictionnaire associant chaque tâche d'ingestion à son statut
jobs: dict[str, dict[str, Any]] = {}

# Ensure upload directory exists
# Crée le répertoire d'upload s'il n'existe pas encore
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
