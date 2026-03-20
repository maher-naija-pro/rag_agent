"""Qdrant vector database — client, store, and collection settings."""

# Permet les annotations de type différées
from __future__ import annotations

# Module pour accéder aux variables d'environnement
import os

# Store vectoriel LangChain pour Qdrant et modes de recherche
from langchain_qdrant import QdrantVectorStore, RetrievalMode
# Client Python officiel pour Qdrant
from qdrant_client import QdrantClient

# Importe les modèles d'embedding dense et sparse
from config.embeddings import EMBEDDINGS, SPARSE_EMBEDDINGS

# S'assure que le fichier .env est chargé
import config.env  # noqa: F401

# Nom de la collection dans Qdrant
COLLECTION     = os.getenv("QDRANT_COLLECTION", "pdf_rag")
# URL du serveur Qdrant
QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
# Clé API pour Qdrant (None si non définie)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

# Variable globale pour le singleton du client Qdrant
_client: QdrantClient | None = None
# Variable globale pour le singleton du store vectoriel
_store: QdrantVectorStore | None = None


def get_client() -> QdrantClient:
    """Lazy singleton for the Qdrant client."""
    # Accède à la variable globale du client
    global _client
    # Vérifie si le client n'est pas encore instancié
    if _client is None:
        # Crée le client Qdrant avec l'URL et la clé API
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # Retourne le client singleton
    return _client


def get_store() -> QdrantVectorStore:
    """Lazy singleton for the hybrid vector store."""
    # Accède à la variable globale du store
    global _store
    # Vérifie si le store n'est pas encore instancié
    if _store is None:
        # Crée le store vectoriel hybride
        _store = QdrantVectorStore(
            # Utilise le client Qdrant singleton
            client=get_client(),
            # Nom de la collection à utiliser
            collection_name=COLLECTION,
            # Modèle d'embedding dense pour la vectorisation
            embedding=EMBEDDINGS,
            # Modèle d'embedding sparse (BM25) pour la recherche hybride
            sparse_embedding=SPARSE_EMBEDDINGS,
            # Mode de recherche hybride (dense + sparse)
            retrieval_mode=RetrievalMode.HYBRID,
        )
    # Retourne le store singleton
    return _store


def set_store(store: QdrantVectorStore) -> None:
    """Replace the global store (called after ingestion)."""
    # Accède à la variable globale du store
    global _store
    # Remplace le store global par le nouveau (utilisé après l'ingestion de documents)
    _store = store
