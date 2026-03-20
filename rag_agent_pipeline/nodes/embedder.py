"""Node 3 — embed_and_store: embed chunks and upsert into Qdrant (hybrid)."""

from __future__ import annotations

# Client Qdrant pour LangChain avec mode de recherche
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from config import (
    # Nom de la collection Qdrant
    COLLECTION,
    # Modèle d'embeddings denses
    EMBEDDINGS,
    # Modèle d'embeddings creux (sparse)
    SPARSE_EMBEDDINGS,
    # Clé API pour l'authentification Qdrant
    QDRANT_API_KEY,
    # URL du serveur Qdrant
    QDRANT_URL,
    # Fonction pour enregistrer le magasin vectoriel globalement
    set_store,
)
# Utilitaires de journalisation
from logger import get_logger, deep_repr
# Type représentant l'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.embedder")


def embed_and_store(state: RAGState) -> dict:
    """Embed chunks (dense + sparse) and upsert into Qdrant for hybrid search."""
    # Journalisation de l'état d'entrée
    log.debug("[INPUT] embed_and_store:\n%s", deep_repr(dict(state)))
    # Récupération des morceaux à encoder depuis l'état
    chunks = state["chunks"]

    # Vérification qu'il y a des morceaux à traiter
    if not chunks:
        log.warning("No chunks to embed")
        # Retour indiquant qu'aucune ingestion n'a eu lieu
        result = {"ingested": False}
        log.debug("[OUTPUT] embed_and_store:\n%s", deep_repr(result))
        return result

    # Journalisation du début de l'encodage
    log.info("Embedding %d chunks (dense + sparse) …", len(chunks))
    try:
        # Création du magasin vectoriel à partir des documents
        store = QdrantVectorStore.from_documents(
            # Liste des morceaux à encoder
            documents=chunks,
            # Modèle d'embeddings denses pour la recherche sémantique
            embedding=EMBEDDINGS,
            # Modèle d'embeddings creux pour la recherche lexicale
            sparse_embedding=SPARSE_EMBEDDINGS,
            # Nom de la collection cible dans Qdrant
            collection_name=COLLECTION,
            # URL du serveur Qdrant
            url=QDRANT_URL,
            # Clé API pour l'authentification
            api_key=QDRANT_API_KEY,
            # Mode hybride combinant recherche dense et creuse
            retrieval_mode=RetrievalMode.HYBRID,
        )
    except Exception as e:
        # Journalisation de l'erreur d'insertion
        log.error("Failed to embed/upsert to Qdrant: %s", e)
        # Propagation de l'exception pour arrêter le pipeline
        raise

    # Enregistrement du magasin vectoriel pour utilisation ultérieure
    set_store(store)
    # Journalisation du succès de l'insertion
    log.info("Upserted to Qdrant at '%s' (hybrid)", QDRANT_URL)
    # Retour indiquant que l'ingestion a réussi
    result = {"ingested": True}
    # Journalisation de l'état de sortie
    log.debug("[OUTPUT] embed_and_store:\n%s", deep_repr(result))
    return result
