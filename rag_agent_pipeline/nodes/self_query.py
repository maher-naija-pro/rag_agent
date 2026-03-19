"""Node — self_query: extract metadata filters from the user's question."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM
from config.pipeline import SELF_QUERY_ENABLED, METADATA_FIELDS
from logger import get_logger
from state import RAGState

log = get_logger("nodes.self_query")

SELF_QUERY_PROMPT = """\
You are a metadata filter extractor for a document search system.
Given a user question, extract any structured filters that can narrow down the search.

Available metadata fields on each document chunk:
{fields_desc}

Rules:
- Output ONLY a JSON object with the extracted filters. No explanation.
- Only include fields you are confident about. Omit uncertain fields.
- If no filters can be extracted, output: {{}}
- For "page", extract the exact page number as an integer.
- For "language", use ISO codes: "fr" for French, "en" for English.
- For "has_tables", use true/false.
- For "dates", extract date strings as they might appear (e.g. "2024-01", "2024-01-15").

Examples:
- "What's on page 5?" → {{"page": 5}}
- "Summarize the French parts" → {{"language": "fr"}}
- "Show me the tables" → {{"has_tables": true}}
- "What happened in January 2024?" → {{"dates": "2024-01"}}
- "What is the main topic?" → {{}}
"""

# Map of metadata fields to their descriptions for the prompt
_FIELD_DESCRIPTIONS = {
    "dates": "dates (list[str]) — date strings found in the chunk",
    "emails": "emails (list[str]) — email addresses found in the chunk",
    "urls": "urls (list[str]) — URLs found in the chunk",
    "keywords": "keywords (list[str]) — top frequent meaningful words",
    "language": 'language (str) — detected language: "fr", "en", or "unknown"',
    "has_tables": "has_tables (bool) — whether the chunk contains table-like structures",
    "char_count": "char_count (int) — character count of the chunk",
    "page": "page (int) — page number in the source PDF",
    "source": "source (str) — PDF filename",
}


def _build_fields_description(fields: list[str]) -> str:
    """Build the fields description for the prompt based on configured METADATA_FIELDS."""
    # Always include page and source (they exist even without metadata extraction)
    all_fields = set(fields) | {"page", "source"}
    lines = []
    for f in sorted(all_fields):
        if f in _FIELD_DESCRIPTIONS:
            lines.append(f"- {_FIELD_DESCRIPTIONS[f]}")
    return "\n".join(lines)


def _parse_filter_response(response: str) -> dict:
    """Parse the LLM JSON response into a filter dict, handling edge cases."""
    text = response.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return {}


def self_query(state: RAGState) -> dict:
    """
    Extract structured metadata filters from the user's question.

    - Analyzes the question for references to pages, dates, language, etc.
    - Stores extracted filters in metadata_filter
    - The retriever node uses these filters to pre-filter Qdrant results
    - Skipped when SELF_QUERY_ENABLED=false
    """
    if not SELF_QUERY_ENABLED:
        log.debug("Self-query disabled — no metadata filters")
        return {"metadata_filter": {}}

    question = state["question"]
    fields_desc = _build_fields_description(METADATA_FIELDS)

    prompt = [
        SystemMessage(content=SELF_QUERY_PROMPT.format(fields_desc=fields_desc)),
        HumanMessage(content=question),
    ]

    try:
        response = LLM.invoke(prompt).content.strip()
        filters = _parse_filter_response(response)

        if filters:
            log.info("Self-query extracted filters: %s", filters)
        else:
            log.debug("Self-query: no filters extracted")

        return {"metadata_filter": filters}

    except Exception as e:
        log.warning("Self-query failed: %s — no filters", e)
        return {"metadata_filter": {}}
