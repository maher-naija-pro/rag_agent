"""Node — rewrite_query: reformulate user question for better retrieval."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM
from config.pipeline import QUERY_REWRITE_ENABLED
from logger import get_logger
from state import RAGState

log = get_logger("nodes.query_rewriter")

REWRITE_PROMPT = """\
You are a query rewriter for a document search system.
Rewrite the user's question to be clear, specific, and optimized for semantic search.

Rules:
- Keep the same language as the original question (French → French, English → English).
- Expand vague references ("it", "that", "this") using conversation history if available.
- Add relevant synonyms or related terms when helpful.
- Do NOT answer the question — only rewrite it.
- Output ONLY the rewritten question, nothing else.
"""


def rewrite_query(state: RAGState) -> dict:
    """
    Reformulate the user's question for better retrieval.

    - Saves original question in original_question
    - Replaces question with a clearer, search-optimized version
    - Skipped when QUERY_REWRITE_ENABLED=false
    """
    question = state["question"]

    if not QUERY_REWRITE_ENABLED:
        log.debug("Query rewriting disabled — passing through")
        return {"original_question": question}

    # Build context from conversation history
    history = state.get("messages", [])
    history_context = ""
    if len(history) > 1:
        recent = history[-4:]  # last 2 turns max
        history_context = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}"
            for m in recent
        )

    user_msg = question
    if history_context:
        user_msg = f"Conversation context:\n{history_context}\n\nQuestion to rewrite:\n{question}"

    prompt = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=user_msg),
    ]

    try:
        rewritten = LLM.invoke(prompt).content.strip()

        # Guard: if LLM returns empty or something clearly wrong, keep original
        if not rewritten or len(rewritten) < 3:
            log.warning("Rewrite returned empty — keeping original")
            return {"original_question": question}

        log.info("Rewrite: '%s' → '%s'", question[:50], rewritten[:50])
        return {
            "original_question": question,
            "question": rewritten,
        }

    except Exception as e:
        log.warning("Query rewrite failed: %s — keeping original", e)
        return {"original_question": question}
