"""Node 2 — chunk: split raw pages into overlapping chunks."""

from __future__ import annotations

# Outil de découpage récursif de texte
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paramètres de taille et chevauchement des morceaux
from config import CHUNK_SIZE, CHUNK_OVERLAP
# Utilitaires de journalisation
from logger import get_logger, deep_repr
# Type représentant l'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.chunker")


def chunk(state: RAGState) -> dict:
    """Split raw pages into overlapping chunks; preserve page metadata."""
    # Journalisation de l'état d'entrée
    log.debug("[INPUT] chunk:\n%s", deep_repr(dict(state)))
    # Récupération des pages brutes depuis l'état
    raw_pages = state["raw_pages"]

    if not raw_pages:
        log.warning("No raw pages to chunk")
        result = {"chunks": []}
        log.debug("[OUTPUT] chunk:\n%s", deep_repr(result))
        return result

    try:
        # Création du découpeur de texte récursif
        splitter = RecursiveCharacterTextSplitter(
            # Taille maximale de chaque morceau en caractères
            chunk_size=CHUNK_SIZE,
            # Nombre de caractères de chevauchement entre morceaux
            chunk_overlap=CHUNK_OVERLAP,
            # Séparateurs utilisés par ordre de priorité
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Découpage des documents en morceaux avec conservation des métadonnées
        chunks = splitter.split_documents(raw_pages)
    except Exception as e:
        log.error("Chunking failed: %s", e)
        result = {"chunks": []}
        log.debug("[OUTPUT] chunk:\n%s", deep_repr(result))
        return result

    # Journalisation du nombre de morceaux créés
    log.info("%d pages → %d chunks", len(raw_pages), len(chunks))
    # Construction du résultat
    result = {"chunks": chunks}
    # Journalisation de l'état de sortie
    log.debug("[OUTPUT] chunk:\n%s", deep_repr(result))
    # Retour des morceaux découpés
    return result
