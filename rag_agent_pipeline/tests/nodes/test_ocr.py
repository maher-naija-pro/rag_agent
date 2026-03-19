"""Tests for nodes.ocr — OCR page extraction."""

from unittest.mock import patch

import fitz


class TestOcrPage:
    @patch("nodes.ocr.pytesseract.image_to_string", return_value="OCR text result")
    def test_returns_ocr_text(self, mock_tesseract, sample_pdf):
        from nodes.ocr import ocr_page

        doc = fitz.open(sample_pdf)
        result = ocr_page(doc[0])
        doc.close()

        assert mock_tesseract.called
        assert result == "OCR text result"

    @patch("nodes.ocr.pytesseract.image_to_string", return_value="   ")
    def test_returns_empty_on_whitespace(self, mock_tesseract, sample_pdf):
        from nodes.ocr import ocr_page

        doc = fitz.open(sample_pdf)
        result = ocr_page(doc[0])
        doc.close()

        assert result == ""

    @patch("nodes.ocr.pytesseract.image_to_string", return_value="Bonjour le monde")
    def test_uses_configured_languages(self, mock_tesseract, sample_pdf):
        from nodes.ocr import ocr_page

        doc = fitz.open(sample_pdf)
        ocr_page(doc[0])
        doc.close()

        call_kwargs = mock_tesseract.call_args
        assert "lang" in call_kwargs.kwargs or len(call_kwargs.args) >= 2
