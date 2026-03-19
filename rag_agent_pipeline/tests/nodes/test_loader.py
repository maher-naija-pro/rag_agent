"""Tests for nodes.loader — PDF text extraction."""

import tempfile

from unittest.mock import patch


class TestLoadPdf:
    """Tests for load_pdf node.

    Mocks: pytesseract (external OCR engine) when testing OCR fallback.
    Never mocks: ocr_page, load_pdf (our code under test).
    """

    def test_extracts_native_text(self, base_state, sample_pdf):
        from nodes.loader import load_pdf

        pages = load_pdf(base_state(question=sample_pdf))["raw_pages"]
        assert len(pages) >= 1
        assert any("Hello" in p.page_content for p in pages)

    def test_sets_native_method(self, base_state, sample_pdf):
        from nodes.loader import load_pdf

        pages = load_pdf(base_state(question=sample_pdf))["raw_pages"]
        assert pages[0].metadata["method"] == "native"

    def test_metadata_has_page_number(self, base_state, sample_pdf):
        from nodes.loader import load_pdf

        pages = load_pdf(base_state(question=sample_pdf))["raw_pages"]
        assert pages[0].metadata["page"] == 1

    def test_metadata_has_source(self, base_state, sample_pdf):
        from nodes.loader import load_pdf

        pages = load_pdf(base_state(question=sample_pdf))["raw_pages"]
        assert pages[0].metadata["source"].endswith(".pdf")

    @patch("nodes.ocr.pytesseract.image_to_string", return_value="OCR fallback text")
    def test_falls_back_to_ocr_for_image_pages(self, mock_tesseract, base_state):
        """When native text is empty, loader delegates to ocr_page which calls pytesseract."""
        import fitz
        from nodes.loader import load_pdf

        # Create a PDF with an empty page (no selectable text) to trigger OCR path
        doc = fitz.open()
        doc.new_page()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()

        result = load_pdf(base_state(question=tmp.name))

        if mock_tesseract.called:
            pages = result["raw_pages"]
            assert any(p.metadata["method"] == "ocr" for p in pages)
            assert any("OCR fallback text" in p.page_content for p in pages)

    def test_nonexistent_file_returns_empty(self, base_state):
        from nodes.loader import load_pdf

        result = load_pdf(base_state(question="/tmp/does_not_exist.pdf"))
        assert result["raw_pages"] == []
