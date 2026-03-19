"""Tests for nodes.ocr — OCR page extraction (real Tesseract)."""

import fitz


class TestOcrPage:
    def test_returns_ocr_text(self, sample_pdf):
        from nodes.ocr import ocr_page

        doc = fitz.open(sample_pdf)
        result = ocr_page(doc[0])
        doc.close()

        # Real Tesseract should extract text from the rendered page
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_empty_on_blank_page(self):
        from nodes.ocr import ocr_page

        # Create a completely blank page (no text, no images)
        doc = fitz.open()
        doc.new_page(width=100, height=100)
        result = ocr_page(doc[0])
        doc.close()

        # Blank page → empty or whitespace-only
        assert result == "" or result.strip() == ""

    def test_extracts_known_text(self):
        """Render a page with known large text and verify OCR reads it."""
        from nodes.ocr import ocr_page

        doc = fitz.open()
        page = doc.new_page()
        # Large font so OCR can reliably read it
        page.insert_text((50, 200), "HELLO WORLD", fontsize=48)
        result = ocr_page(page)
        doc.close()

        assert "HELLO" in result.upper() or "WORLD" in result.upper()
