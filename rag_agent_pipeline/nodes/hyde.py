"""Node — hyde: generate a hypothetical document for better retrieval."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM
from config.pipeline import HYDE_ENABLED
from logger import get_logger
from state import RAGState

log = get_logger("nodes.hyde")

HYDE_PROMPT = """\
You are a document passage generator.
Given a question, write a short paragraph (3-5 sentences) that would appear in a document \
answering this question. Write it as if you are quoting from an actual document.

Rules:
- Keep the same language as the question (French → French, English → English).
- Write in a formal, document-like style (not conversational).
- Do NOT say "the document says" — write as if you ARE the document.
- Output ONLY the paragraph, nothing else.
"""


def hyde(state: RAGState) -> dict:
    """
    Generate a hypothetical document passage and add it to expanded_queries.

    The fake answer's embedding is closer to real document chunks
    than the original question, improving retrieval for knowledge-heavy queries.

    - Skipped when HYDE_ENABLED=false (default)
    - Appends the hypothetical passage to existing expanded_queries
    - On failure, passes through without modification
    """
    if not HYDE_ENABLED:
        log.debug("HyDE disabled — passing through")
        return {}

    question = state["question"]
    existing_queries = state.get("expanded_queries", [question])

    prompt = [
        SystemMessage(content=HYDE_PROMPT),
        HumanMessage(content=question),
    ]

    try:
        hypothetical = LLM.invoke(prompt).content.strip()

        if not hypothetical or len(hypothetical) < 10:
            log.warning("HyDE returned empty — skipping")
            return {}

        updated_queries = existing_queries + [hypothetical]

        log.info("HyDE: generated %d-char hypothetical passage", len(hypothetical))
        return {"expanded_queries": updated_queries}

    except Exception as e:
        log.warning("HyDE failed: %s — skipping", e)
        return {}
