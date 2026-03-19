"""Node — expand_query: generate variant queries for broader retrieval."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM
from config.pipeline import QUERY_EXPANSION_ENABLED, QUERY_EXPANSION_COUNT
from logger import get_logger
from state import RAGState

log = get_logger("nodes.query_expander")

EXPAND_PROMPT = """\
You are a search query generator for a document retrieval system.
Given a question, generate {count} alternative search queries that cover different angles, \
synonyms, or phrasings to find relevant document passages.

Rules:
- Keep the same language as the original question.
- Each variant should use different keywords or phrasing.
- Do NOT answer the question — only generate search queries.
- Output one query per line, no numbering, no bullet points.
"""


def expand_query(state: RAGState) -> dict:
    """
    Generate variant queries for broader retrieval coverage.

    - Takes the (already rewritten) question
    - Generates QUERY_EXPANSION_COUNT variants via LLM
    - Stores [original + variants] in expanded_queries
    - Skipped when QUERY_EXPANSION_ENABLED=false
    """
    question = state["question"]

    if not QUERY_EXPANSION_ENABLED:
        log.debug("Query expansion disabled — single query")
        return {"expanded_queries": [question]}

    prompt = [
        SystemMessage(content=EXPAND_PROMPT.format(count=QUERY_EXPANSION_COUNT)),
        HumanMessage(content=question),
    ]

    try:
        response = LLM.invoke(prompt).content.strip()
        variants = [line.strip() for line in response.split("\n") if line.strip()]

        # Filter out empty or too-short variants
        variants = [v for v in variants if len(v) >= 5]

        # Limit to requested count
        variants = variants[:QUERY_EXPANSION_COUNT]

        # Always include the original query
        all_queries = [question] + [v for v in variants if v != question]

        log.info("Expanded '%s' → %d queries", question[:40], len(all_queries))
        return {"expanded_queries": all_queries}

    except Exception as e:
        log.warning("Query expansion failed: %s — using single query", e)
        return {"expanded_queries": [question]}
