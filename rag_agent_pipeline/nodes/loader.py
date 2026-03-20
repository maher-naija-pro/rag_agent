"""Node 1 — load_pdf: extract text from every PDF page (native or OCR)."""

from __future__ import annotations

# Gestion des chemins de fichiers de manière portable
from pathlib import Path

# PyMuPDF — bibliothèque de manipulation de PDF
import fitz
# Classe représentant un document avec contenu et métadonnées
from langchain_core.documents import Document

# Utilitaires de journalisation
from logger import get_logger, deep_repr
# Fonction OCR pour les pages sans texte natif
from nodes.ocr import ocr_page
# Type représentant l'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.loader")


def load_pdf(state: RAGState) -> dict:
    """
    Read every PDF page.
    - Native text pages  → extracted directly via PyMuPDF (fast).
    - Image-only pages   → delegated to nodes.ocr for Tesseract OCR.

    Handles corrupted, encrypted, and empty PDFs gracefully.
    """
    # Journalisation de l'état d'entrée
    log.debug("[INPUT] load_pdf:\n%s", deep_repr(dict(state)))
    # Récupération du chemin du fichier PDF depuis l'état
    pdf_path = state["question"]
    # Liste pour stocker les pages extraites
    pages: list[Document] = []

    # Vérification que le fichier existe sur le disque
    if not Path(pdf_path).is_file():
        log.error("File not found: %s", pdf_path)
        result = {"raw_pages": []}
        log.debug("[OUTPUT] load_pdf:\n%s", deep_repr(result))
        return result

    # Ouverture du PDF avec gestion des erreurs pour les fichiers corrompus ou chiffrés
    try:
        # Ouverture du document PDF
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.error("Failed to open PDF '%s': %s", pdf_path, e)
        result = {"raw_pages": []}
        log.debug("[OUTPUT] load_pdf:\n%s", deep_repr(result))
        return result

    # Vérification si le PDF est protégé par mot de passe
    if doc.is_encrypted:
        # Avertissement PDF chiffré
        log.warning("PDF is password-protected: %s — attempting without password", pdf_path)
        # Tentative d'authentification avec un mot de passe vide
        if not doc.authenticate(""):
            log.error("Cannot open encrypted PDF '%s' — password required", pdf_path)
            doc.close()
            result = {"raw_pages": []}
            log.debug("[OUTPUT] load_pdf:\n%s", deep_repr(result))
            return result

    # Extraction du nom du fichier comme identifiant source
    source_name = Path(pdf_path).name

    # Parcours de chaque page du PDF
    for i, page in enumerate(doc):
        text = ""
        # Méthode d'extraction par défaut : texte natif
        method = "native"

        # Tentative d'extraction du texte natif en premier
        try:
            # Extraction du texte intégré dans le PDF
            text = page.get_text("text").strip()
        except Exception as e:
            log.warning("Native text extraction failed on page %d: %s", i + 1, e)

        # Repli sur l'OCR pour les pages contenant uniquement des images
        if not text:
            # Changement de méthode vers OCR
            method = "ocr"
            try:
                # Appel de la fonction OCR sur la page
                text = ocr_page(page)
            except Exception as e:
                log.warning("OCR failed on page %d: %s", i + 1, e)
                # Passage à la page suivante en cas d'échec OCR
                continue

        # Vérification que du texte a été extrait
        if text.strip():
            pages.append(Document(
                # Contenu textuel de la page
                page_content=text,
                # Métadonnées associées
                metadata={"source": source_name, "page": i + 1, "method": method},
            ))

    # Nombre total de pages dans le PDF
    total_pages = len(doc)
    # Fermeture du document PDF
    doc.close()

    if not pages:
        log.warning("No text extracted from '%s' (%d pages in PDF)", source_name, total_pages)

    # Comptage des pages extraites par OCR
    ocr_n = sum(1 for p in pages if p.metadata["method"] == "ocr")
    # Journalisation du résumé d'extraction
    log.info("%d pages extracted (%d via OCR)", len(pages), ocr_n)
    # Construction du résultat
    result = {"raw_pages": pages, "source": source_name}
    # Journalisation de l'état de sortie
    log.debug("[OUTPUT] load_pdf:\n%s", deep_repr(result))
    # Retour des pages extraites
    return result
