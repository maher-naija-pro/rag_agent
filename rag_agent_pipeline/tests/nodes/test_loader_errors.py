"""Tests for loader error handling — real files, no mocks."""

import os
import tempfile

import fitz


class TestLoadPdfErrors:
    """Test error paths in load_pdf using real corrupted/encrypted/empty PDFs."""

    def test_nonexistent_file(self, base_state):
        from nodes.loader import load_pdf

        result = load_pdf(base_state(question="/tmp/surely_does_not_exist_abc123.pdf"))
        assert result["raw_pages"] == []

    def test_corrupted_file(self, base_state):
        """A file that exists but is not a valid PDF."""
        from nodes.loader import load_pdf

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(b"this is not a PDF at all")
        tmp.close()

        try:
            result = load_pdf(base_state(question=tmp.name))
            assert result["raw_pages"] == []
        finally:
            os.unlink(tmp.name)

    def test_truncated_pdf(self, base_state):
        """A file with a PDF header but truncated content."""
        from nodes.loader import load_pdf

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(b"%PDF-1.4\n%%EOF")
        tmp.close()

        try:
            result = load_pdf(base_state(question=tmp.name))
            assert result["raw_pages"] == []
        finally:
            os.unlink(tmp.name)

    def test_encrypted_pdf_no_password(self, base_state):
        """A password-protected PDF that can't be opened without credentials."""
        from nodes.loader import load_pdf

        doc = fitz.open()
        doc.new_page()
        page = doc[0]
        page.insert_text((50, 72), "Secret content")

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        # Save with encryption — owner and user password
        doc.save(
            tmp.name,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner123",
            user_pw="user123",
            permissions=fitz.PDF_PERM_ACCESSIBILITY,
        )
        doc.close()

        try:
            result = load_pdf(base_state(question=tmp.name))
            # Should return empty — can't authenticate with empty password
            assert result["raw_pages"] == []
        finally:
            os.unlink(tmp.name)

    def test_multi_page_pdf(self, base_state):
        """Verify all pages are extracted from a multi-page PDF."""
        from nodes.loader import load_pdf

        doc = fitz.open()
        for i in range(5):
            page = doc.new_page()
            page.insert_text((50, 72), f"Page {i + 1} content")

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()

        try:
            result = load_pdf(base_state(question=tmp.name))
            assert len(result["raw_pages"]) == 5
            assert result["raw_pages"][0].metadata["page"] == 1
            assert result["raw_pages"][4].metadata["page"] == 5
        finally:
            os.unlink(tmp.name)

    def test_sets_source_field(self, base_state):
        """Verify loader returns source filename in state."""
        from nodes.loader import load_pdf

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "Test")

        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, prefix="myfile_")
        doc.save(tmp.name)
        doc.close()

        try:
            result = load_pdf(base_state(question=tmp.name))
            assert "source" in result
            assert result["source"].endswith(".pdf")
        finally:
            os.unlink(tmp.name)
