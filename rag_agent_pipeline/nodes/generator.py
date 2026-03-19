"""Node 5 — generate: stream an answer from the LLM."""

from __future__ import annotations

import sys

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import LLM
from logger import get_logger
from state import RAGState

log = get_logger("nodes.generator")

SYSTEM_TEMPLATE = """\
You are a precise, concise assistant. Answer only from the document excerpts below.

Rules:
- Be SHORT and DIRECT. Answer the question in 1-3 sentences maximum.
- Reply in the SAME language as the user's question (French → French, English → English).
- Cite the page number(s), e.g. [page 3].
- If the answer is absent, say so briefly — never fabricate.
- Do NOT repeat or summarize the entire document. Only give what was asked.

CONTEXT:
{context}
"""

NO_CONTEXT_ANSWER = (
    "I could not find any relevant information in the document to answer this question. "
    "Try rephrasing your question, or check that the document covers this topic."
)


def _format_docs(docs: list[Document]) -> str:
    """Render a list of documents into a single context string with page citations."""
    return "\n\n---\n\n".join(
        f"[page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )


def generate(state: RAGState) -> dict:
    """
    Stream an answer from the LLM using:
      - System prompt  containing the reranked context
      - History        from state["messages"] (populated by InMemorySaver)
      - Latest question

    If no context documents were retrieved, returns a fixed refusal
    instead of calling the LLM (prevents hallucination).
    """
    context_docs = state.get("context", [])

    # Guard: no context → refuse to answer instead of hallucinating
    if not context_docs:
        log.warning("No context documents — returning refusal")
        return {
            "answer":   NO_CONTEXT_ANSWER,
            "messages": [AIMessage(content=NO_CONTEXT_ANSWER)],
        }

    context_str = _format_docs(context_docs)
    history     = state["messages"][:-1]

    prompt = (
        [SystemMessage(content=SYSTEM_TEMPLATE.format(context=context_str))]
        + history
        + [HumanMessage(content=state["question"])]
    )

    log.info("Streaming answer …")
    parts: list[str] = []
    try:
        sys.stdout.write("Assistant: ")
        for chunk_token in LLM.stream(prompt):
            tok = chunk_token.content
            sys.stdout.write(tok)
            sys.stdout.flush()
            parts.append(tok)
        sys.stdout.write("\n")
    except Exception as e:
        sys.stdout.write("\n")
        log.error("LLM streaming failed: %s", e)
        error_msg = "An error occurred while generating the answer. Please try again."
        return {
            "answer":   error_msg,
            "messages": [AIMessage(content=error_msg)],
        }

    answer = "".join(parts)
    log.info("Answer generated (%d chars)", len(answer))

    return {
        "answer":   answer,
        "messages": [AIMessage(content=answer)],
    }
