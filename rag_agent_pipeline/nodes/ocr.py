"""OCR utilities — convert image-only PDF pages to text via Tesseract."""

from __future__ import annotations

import io

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from config import OCR_DPI, OCR_LANGUAGES
from logger import get_logger

log = get_logger("nodes.ocr")


def ocr_page(page: fitz.Page) -> str:
    """Render a single PDF page to image and run Tesseract OCR."""
    scale = OCR_DPI / 72
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img, lang=OCR_LANGUAGES)
    result = text.strip()
    if result:
        log.debug("OCR extracted %d characters", len(result))
    else:
        log.warning("OCR returned empty text")
    return result
