"""Node 3 — extract_metadata: enrich chunks with structured metadata."""

from __future__ import annotations

import re
from collections import Counter

from langchain_core.documents import Document

from config.pipeline import METADATA_EXTRACTION_ENABLED, METADATA_FIELDS
from logger import get_logger
from state import RAGState

log = get_logger("nodes.metadata")

# ── Regex patterns for metadata extraction ───────────────────────────────────

_DATE_PATTERNS = [
    # ISO: 2024-01-15
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    # EU: 15/01/2024 or 15-01-2024
    re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b"),
    # Written: January 15, 2024 / 15 January 2024 / Jan 2024
    re.compile(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Oct|Nov|Dec)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # French: 15 janvier 2024
    re.compile(
        r"\b(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|"
        r"août|septembre|octobre|novembre|décembre)\s+\d{4})\b",
        re.IGNORECASE,
    ),
]

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")

_PHONE_PATTERN = re.compile(
    r"\b(?:\+\d{1,3}[\s\-]?)?"
    r"(?:\(?\d{1,4}\)?[\s\-]?)?"
    r"\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{2,4}\b"
)

_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")


def _extract_dates(text: str) -> list[str]:
    """Extract date strings from text."""
    dates = []
    for pattern in _DATE_PATTERNS:
        dates.extend(pattern.findall(text))
    return sorted(set(dates))


def _extract_emails(text: str) -> list[str]:
    """Extract email addresses from text."""
    return sorted(set(_EMAIL_PATTERN.findall(text)))


def _extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    return sorted(set(_URL_PATTERN.findall(text)))


def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top-N most frequent meaningful words (>4 chars) as keywords."""
    words = re.findall(r"\b[a-zA-ZÀ-ÿ]{5,}\b", text.lower())
    # Filter common stop words
    stop = {
        "about", "after", "again", "being", "below", "between", "could",
        "didn", "doing", "during", "every", "first", "found", "given",
        "great", "having", "their", "there", "these", "thing", "think",
        "those", "three", "through", "under", "until", "using", "value",
        "where", "which", "while", "would", "would", "years",
        "cette", "comme", "dans", "elles", "entre", "leurs", "mais",
        "notre", "nous", "notre", "autres", "aussi", "avant", "avoir",
        "cette", "comme", "depuis", "encore", "entre", "faire", "leurs",
        "même", "notre", "notre", "peut", "plus", "pour", "quand",
        "quelque", "sont", "tout", "très", "votre",
    }
    filtered = [w for w in words if w not in stop]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def _detect_language(text: str) -> str:
    """Simple language detection based on common word frequency."""
    sample = text[:2000].lower()

    fr_markers = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "en", "que", "pour", "dans", "sur"}
    en_markers = {"the", "is", "are", "was", "were", "of", "and", "to", "in", "for", "that", "with", "on", "at"}

    words = set(re.findall(r"\b\w+\b", sample))
    fr_score = len(words & fr_markers)
    en_score = len(words & en_markers)

    if fr_score > en_score:
        return "fr"
    if en_score > fr_score:
        return "en"
    return "unknown"


def _has_tables(text: str) -> bool:
    """Detect if text likely contains table-like structures."""
    lines = text.split("\n")
    tabular_lines = sum(1 for line in lines if line.count("|") >= 2 or line.count("\t") >= 2)
    return tabular_lines >= 3


def _extract_for_chunk(doc: Document, fields: set[str]) -> dict:
    """Extract requested metadata fields for a single chunk."""
    text = doc.page_content
    meta = {}

    if "dates" in fields:
        dates = _extract_dates(text)
        if dates:
            meta["dates"] = dates

    if "emails" in fields:
        emails = _extract_emails(text)
        if emails:
            meta["emails"] = emails

    if "urls" in fields:
        urls = _extract_urls(text)
        if urls:
            meta["urls"] = urls

    if "keywords" in fields:
        keywords = _extract_keywords(text)
        if keywords:
            meta["keywords"] = keywords

    if "language" in fields:
        meta["language"] = _detect_language(text)

    if "has_tables" in fields:
        meta["has_tables"] = _has_tables(text)

    if "char_count" in fields:
        meta["char_count"] = len(text)

    return meta


def extract_metadata(state: RAGState) -> dict:
    """
    Enrich each chunk with structured metadata extracted from its content.

    Controlled by METADATA_FIELDS env var (comma-separated).
    Available fields: dates, emails, urls, keywords, language, has_tables, char_count.
    """
    chunks = state.get("chunks", [])

    if not METADATA_EXTRACTION_ENABLED:
        log.debug("Metadata extraction disabled — passing through")
        return {"chunks": chunks}

    if not chunks:
        log.warning("No chunks to extract metadata from")
        return {"chunks": []}

    fields = set(METADATA_FIELDS)
    if not fields:
        log.info("METADATA_FIELDS is empty — skipping metadata extraction")
        return {"chunks": chunks}

    enriched = []
    stats: dict[str, int] = {f: 0 for f in fields}

    for doc in chunks:
        extracted = _extract_for_chunk(doc, fields)

        new_meta = {**doc.metadata, **extracted}
        enriched.append(Document(
            page_content=doc.page_content,
            metadata=new_meta,
        ))

        for key in extracted:
            stats[key] = stats.get(key, 0) + 1

    log.info(
        "Metadata extracted for %d chunks: %s",
        len(enriched),
        ", ".join(f"{k}={v}" for k, v in stats.items() if v > 0),
    )
    return {"chunks": enriched}
