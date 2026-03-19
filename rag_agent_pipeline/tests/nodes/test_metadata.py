"""Tests for nodes.metadata — metadata extraction from chunks."""

import sys

from langchain_core.documents import Document

import nodes.metadata
_mod = sys.modules["nodes.metadata"]


class TestExtractMetadata:
    """Tests for the extract_metadata() node function — pure logic, no external services."""

    def test_extracts_dates(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", ["dates", "keywords", "language"])

        chunks = [Document(page_content="Report from 2024-01-15 and 2023-12-01.", metadata={"page": 1})]
        result = _mod.extract_metadata(base_state(chunks=chunks))

        assert len(result["chunks"]) == 1
        meta = result["chunks"][0].metadata
        assert "dates" in meta
        assert "2024-01-15" in meta["dates"]

    def test_extracts_french_dates(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", ["dates"])

        chunks = [Document(page_content="Le 15 janvier 2024 nous avons signé.", metadata={"page": 1})]
        result = _mod.extract_metadata(base_state(chunks=chunks))

        meta = result["chunks"][0].metadata
        assert "dates" in meta

    def test_extracts_keywords(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", ["keywords"])

        chunks = [Document(
            page_content="Machine learning pipeline processes documents efficiently. "
                         "Pipeline architecture handles document processing.",
            metadata={"page": 1},
        )]
        result = _mod.extract_metadata(base_state(chunks=chunks))

        meta = result["chunks"][0].metadata
        assert "keywords" in meta
        assert isinstance(meta["keywords"], list)
        assert len(meta["keywords"]) > 0

    def test_detects_language_french(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", ["language"])

        chunks = [Document(
            page_content="Le document est rédigé en français pour les utilisateurs dans le cadre du projet.",
            metadata={"page": 1},
        )]
        result = _mod.extract_metadata(base_state(chunks=chunks))
        assert result["chunks"][0].metadata["language"] == "fr"

    def test_detects_language_english(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", ["language"])

        chunks = [Document(
            page_content="The document is written in English for the users of the project.",
            metadata={"page": 1},
        )]
        result = _mod.extract_metadata(base_state(chunks=chunks))
        assert result["chunks"][0].metadata["language"] == "en"

    def test_preserves_existing_metadata(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", ["dates"])

        chunks = [Document(page_content="Hello world.", metadata={"page": 3, "source": "test.pdf"})]
        result = _mod.extract_metadata(base_state(chunks=chunks))

        meta = result["chunks"][0].metadata
        assert meta["page"] == 3
        assert meta["source"] == "test.pdf"

    def test_empty_chunks_returns_empty(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)

        result = _mod.extract_metadata(base_state(chunks=[]))
        assert result["chunks"] == []

    def test_no_dates_omits_key(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", ["dates"])

        chunks = [Document(page_content="No dates here at all.", metadata={"page": 1})]
        result = _mod.extract_metadata(base_state(chunks=chunks))
        meta = result["chunks"][0].metadata
        assert "dates" not in meta

    def test_disabled_passes_through(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", False)

        chunks = [Document(page_content="2024-01-01", metadata={"page": 1})]
        result = _mod.extract_metadata(base_state(chunks=chunks))
        assert result["chunks"][0].metadata == {"page": 1}

    def test_all_extra_fields(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", ["emails", "urls", "has_tables", "char_count"])

        chunks = [Document(
            page_content="Contact info@example.com or visit https://test.com\n| a | b |\n| c | d |\n| e | f |",
            metadata={"page": 1},
        )]
        result = _mod.extract_metadata(base_state(chunks=chunks))
        meta = result["chunks"][0].metadata

        assert "emails" in meta
        assert "info@example.com" in meta["emails"]
        assert "urls" in meta
        assert "https://test.com" in meta["urls"]
        assert meta["has_tables"] is True
        assert meta["char_count"] > 0

    def test_empty_fields_skips_extraction(self, base_state, monkeypatch):
        monkeypatch.setattr(_mod, "METADATA_EXTRACTION_ENABLED", True)
        monkeypatch.setattr(_mod, "METADATA_FIELDS", [])

        chunks = [Document(page_content="Some text 2024-01-01", metadata={"page": 1})]
        result = _mod.extract_metadata(base_state(chunks=chunks))
        assert "dates" not in result["chunks"][0].metadata


class TestExtractDates:
    def test_iso_format(self):
        assert "2024-01-15" in _mod._extract_dates("Date: 2024-01-15")

    def test_eu_format(self):
        assert "15/01/2024" in _mod._extract_dates("Date: 15/01/2024")

    def test_written_english(self):
        assert len(_mod._extract_dates("January 15, 2024")) >= 1

    def test_no_dates(self):
        assert _mod._extract_dates("No dates here") == []


class TestExtractEmails:
    def test_finds_emails(self):
        result = _mod._extract_emails("Contact us at info@example.com or support@test.org")
        assert "info@example.com" in result
        assert "support@test.org" in result

    def test_no_emails(self):
        assert _mod._extract_emails("No emails here") == []


class TestExtractUrls:
    def test_finds_urls(self):
        result = _mod._extract_urls("Visit https://example.com and http://test.org/page")
        assert "https://example.com" in result

    def test_no_urls(self):
        assert _mod._extract_urls("No links here") == []


class TestExtractKeywords:
    def test_returns_top_n(self):
        text = "pipeline " * 10 + "document " * 8 + "processing " * 6
        result = _mod._extract_keywords(text, top_n=3)
        assert len(result) <= 3
        assert "pipeline" in result

    def test_filters_short_words(self):
        assert _mod._extract_keywords("the a is of and or but not", top_n=5) == []


class TestDetectLanguage:
    def test_french(self):
        assert _mod._detect_language("le document est dans la base de données") == "fr"

    def test_english(self):
        assert _mod._detect_language("the document is in the database for processing") == "en"


class TestHasTables:
    def test_detects_pipe_tables(self):
        assert _mod._has_tables("| col1 | col2 |\n| a | b |\n| c | d |\n| e | f |") is True

    def test_detects_tab_tables(self):
        assert _mod._has_tables("a\tb\tc\nd\te\tf\ng\th\ti\nj\tk\tl") is True

    def test_no_tables(self):
        assert _mod._has_tables("Just regular text.") is False
