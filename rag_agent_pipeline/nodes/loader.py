"""Node 1 — load_pdf: extract text from every PDF page (native or OCR)."""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from logger import get_logger
from nodes.ocr import ocr_page
from state import RAGState

log = get_logger("nodes.loader")


def load_pdf(state: RAGState) -> dict:
    """
    Read every PDF page.
    - Native text pages  → extracted directly via PyMuPDF (fast).
    - Image-only pages   → delegated to nodes.ocr for Tesseract OCR.

    Handles corrupted, encrypted, and empty PDFs gracefully.
    """
    pdf_path = state["question"]
    pages: list[Document] = []

    # Validate the file exists
    if not Path(pdf_path).is_file():
        log.error("File not found: %s", pdf_path)
        return {"raw_pages": []}

    # Open with error handling for corrupted / encrypted PDFs
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.error("Failed to open PDF '%s': %s", pdf_path, e)
        return {"raw_pages": []}

    # Check for encryption / password protection
    if doc.is_encrypted:
        log.warning("PDF is password-protected: %s — attempting without password", pdf_path)
        if not doc.authenticate(""):
            log.error("Cannot open encrypted PDF '%s' — password required", pdf_path)
            doc.close()
            return {"raw_pages": []}

    source_name = Path(pdf_path).name

    for i, page in enumerate(doc):
        text = ""
        method = "native"

        # Try native text extraction first
        try:
            text = page.get_text("text").strip()
        except Exception as e:
            log.warning("Native text extraction failed on page %d: %s", i + 1, e)

        # Fall back to OCR for image-only pages
        if not text:
            method = "ocr"
            try:
                text = ocr_page(page)
            except Exception as e:
                log.warning("OCR failed on page %d: %s", i + 1, e)
                continue

        if text.strip():
            pages.append(Document(
                page_content=text,
                metadata={"source": source_name, "page": i + 1, "method": method},
            ))

    total_pages = len(doc)
    doc.close()

    if not pages:
        log.warning("No text extracted from '%s' (%d pages in PDF)", source_name, total_pages)

    ocr_n = sum(1 for p in pages if p.metadata["method"] == "ocr")
    log.info("%d pages extracted (%d via OCR)", len(pages), ocr_n)
    return {"raw_pages": pages, "source": source_name}
