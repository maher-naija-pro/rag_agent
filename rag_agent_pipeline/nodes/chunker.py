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

        # Enrichit chaque chunk avec line_start et line_end.
        # On indexe le texte original de chaque page, puis on localise
        # chaque chunk dedans pour calculer les numéros de ligne.
        page_texts: dict[int, str] = {}
        page_search_offset: dict[int, int] = {}
        for p in raw_pages:
            pg = p.metadata.get("page", 0)
            if pg not in page_texts:
                page_texts[pg] = p.page_content
                page_search_offset[pg] = 0

        for c in chunks:
            pg = c.metadata.get("page", 0)
            full_text = page_texts.get(pg, "")
            # Cherche à partir du dernier offset pour gérer les doublons
            start_from = page_search_offset.get(pg, 0)
            # Utilise les 60 premiers chars (assez pour un match unique)
            needle = c.page_content[:60].strip()
            idx = full_text.find(needle, start_from) if needle else -1
            if idx >= 0:
                line_start = full_text[:idx].count("\n") + 1
                line_end = line_start + c.page_content.count("\n")
                # Avance l'offset pour le prochain chunk de la même page
                page_search_offset[pg] = idx + len(needle)
            else:
                line_start = 1
                line_end = 1 + c.page_content.count("\n")
            c.metadata["line_start"] = line_start
            c.metadata["line_end"] = line_end

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
