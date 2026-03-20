"""OCR utilities — convert image-only PDF pages to text via Tesseract."""

from __future__ import annotations

# Module pour manipuler les flux d'octets en mémoire
import io

# PyMuPDF — bibliothèque de manipulation de PDF
import fitz
# Interface Python pour le moteur OCR Tesseract
import pytesseract
# Bibliothèque de traitement d'images
from PIL import Image

# Paramètres de résolution et langues OCR
from config import OCR_DPI, OCR_LANGUAGES
# Utilitaire de journalisation
from logger import get_logger

# Initialisation du logger pour ce module
log = get_logger("nodes.ocr")


def ocr_page(page: fitz.Page) -> str:
    """Render a single PDF page to image and run Tesseract OCR."""
    # Calcul du facteur d'échelle à partir de la résolution souhaitée (72 DPI par défaut pour un PDF)
    scale = OCR_DPI / 72
    # Création de la matrice de transformation pour le rendu
    mat = fitz.Matrix(scale, scale)
    # Rendu de la page PDF en image pixelisée sans canal alpha
    pix = page.get_pixmap(matrix=mat, alpha=False)
    # Conversion des octets PNG en objet Image PIL
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    # Exécution de l'OCR Tesseract sur l'image
    text = pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
    # Suppression des espaces en début et fin de texte
    result = text.strip()
    if result:
        # Journalisation du nombre de caractères extraits
        log.debug("OCR extracted %d characters", len(result))
    else:
        # Avertissement si aucun texte n'a été extrait
        log.warning("OCR returned empty text")
    # Retour du texte reconnu par OCR
    return result
