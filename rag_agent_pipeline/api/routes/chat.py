"""POST /api/chat        — replay-streamed answer (SSE).
POST /api/chat/stream — true token-by-token LLM streaming (SSE).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from api.schemas import ChatRequest
from api.state import graph, sessions, checkpointer
from config import (
    LLM,
    RETRIEVAL_K,
    HYBRID_FUSION_ALPHA,
    SIMILARITY_THRESHOLD,
    get_store,
)
from nodes.reranker import _build_reranker, _filter_by_score
from config import RERANK_SCORE_THRESHOLD  # noqa: E402
from logger import get_logger
from nodes.generator import _format_docs, SYSTEM_TEMPLATE

router = APIRouter()
log = get_logger("api.routes.chat")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ── POST /api/chat (chunked replay) ─────────────────────────────────────────

@router.post("/api/chat")
async def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Ingest a PDF first.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    thread_id = session["thread_id"]
    # Use the on-disk filename (UUID-based) that matches metadata.source in Qdrant
    qdrant_source = Path(session.get("file_path", "")).name if session.get("file_path") else ""

    log.info("Chat request (session=%s): '%s'", req.session_id, req.question[:80])

    async def event_stream():
        try:
            result = graph.invoke(
                {
                    "messages":  [HumanMessage(content=req.question)],
                    "question":  req.question,
                    "source":    qdrant_source,
                    "raw_pages": [], "chunks": [], "candidates": [], "context": [],
                    "answer": "", "ingested": True,
                },
                config={"configurable": {"thread_id": thread_id}},
            )

            context_docs = result.get("context", [])
            source_pages = sorted(set(
                d.metadata.get("page", 0) for d in context_docs if d.metadata.get("page")
            ))
            answer = result.get("answer", "")

            chunk_size = 4
            for i in range(0, len(answer), chunk_size):
                yield f"data: {json.dumps({'type': 'token', 'content': answer[i:i+chunk_size]})}\n\n"
                await asyncio.sleep(0.01)

            if source_pages:
                yield f"data: {json.dumps({'type': 'sources', 'pages': source_pages})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'answer': answer})}\n\n"

        except Exception as e:
            log.error("Chat error (session=%s): %s", req.session_id, e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


# ── POST /api/chat/stream (true token streaming) ────────────────────────────

@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Ingest a PDF first.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    thread_id = session["thread_id"]
    qdrant_source = Path(session.get("file_path", "")).name if session.get("file_path") else ""
    config    = {"configurable": {"thread_id": thread_id}}

    log.info("Chat stream request (session=%s): '%s'", req.session_id, req.question[:80])

    async def event_stream():
        try:
            # Stage 0 — Rewrite query for better retrieval
            from nodes.query_rewriter import rewrite_query
            rewrite_state = {
                "question": req.question,
                "original_question": "",
                "messages": [],
                "source": qdrant_source,
            }
            # Load conversation history from checkpoint
            checkpoint = checkpointer.get(config)
            if checkpoint and "channel_values" in checkpoint:
                rewrite_state["messages"] = checkpoint["channel_values"].get("messages", [])

            rewrite_result = rewrite_query(rewrite_state)
            search_question = rewrite_result.get("question", req.question)

            # Stage 1 — Retrieve candidates from Qdrant
            search_kwargs: dict = {"k": RETRIEVAL_K}
            if SIMILARITY_THRESHOLD > 0.0:
                search_kwargs["score_threshold"] = SIMILARITY_THRESHOLD
            if HYBRID_FUSION_ALPHA != 0.5:
                search_kwargs["alpha"] = HYBRID_FUSION_ALPHA

            base_retriever = get_store().as_retriever(
                search_type="similarity", search_kwargs=search_kwargs,
            )
            candidates = base_retriever.invoke(search_question)

            # Stage 2 — Rerank candidates
            reranker = _build_reranker()
            context_docs = list(reranker.compress_documents(candidates, search_question))
            context_docs = _filter_by_score(context_docs, RERANK_SCORE_THRESHOLD)

            source_pages = sorted(set(
                d.metadata.get("page", 0) for d in context_docs if d.metadata.get("page")
            ))

            # Stage 3 — Generate answer via LLM streaming
            context_str = _format_docs(context_docs)
            checkpoint  = checkpointer.get(config)
            history     = []
            if checkpoint and "channel_values" in checkpoint:
                msgs = checkpoint["channel_values"].get("messages", [])
                history = msgs[:-1] if msgs else []

            prompt = (
                [SystemMessage(content=SYSTEM_TEMPLATE.format(context=context_str))]
                + history
                + [HumanMessage(content=req.question)]
            )

            full_answer = ""
            for chunk_token in LLM.stream(prompt):
                tok = chunk_token.content
                if tok:
                    full_answer += tok
                    yield f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"

            graph.invoke(
                {
                    "messages":  [HumanMessage(content=req.question)],
                    "question":  req.question,
                    "source":    qdrant_source,
                    "raw_pages": [], "chunks": [], "candidates": [],
                    "context": context_docs, "answer": full_answer, "ingested": True,
                },
                config=config,
            )

            if source_pages:
                yield f"data: {json.dumps({'type': 'sources', 'pages': source_pages})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer})}\n\n"

        except Exception as e:
            log.error("Chat stream error (session=%s): %s", req.session_id, e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)
