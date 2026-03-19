"""POST /api/chat — true token-by-token LLM streaming (SSE)."""

from __future__ import annotations

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


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream answer tokens in real time via SSE.

    Pipeline stages executed sequentially:
      1. Query rewriting (improve retrieval semantics)
      2. Retrieval from Qdrant (hybrid dense + sparse search)
      3. Reranking candidates
      4. LLM streaming — tokens yielded as they are generated
      5. Graph invocation to persist conversation in checkpointer
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Ingest a PDF first.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    thread_id = session["thread_id"]
    qdrant_source = (
        Path(session.get("file_path", "")).name
        if session.get("file_path")
        else ""
    )
    config = {"configurable": {"thread_id": thread_id}}

    log.info("Chat request (session=%s): '%s'", req.session_id, req.question[:80])

    async def event_stream():
        try:
            # ── Stage 1: Rewrite query for better retrieval ──────────────
            from nodes.query_rewriter import rewrite_query

            rewrite_state: dict = {
                "question": req.question,
                "original_question": "",
                "messages": [],
                "source": qdrant_source,
            }
            # Load conversation history from the checkpoint
            checkpoint = checkpointer.get(config)
            if checkpoint and "channel_values" in checkpoint:
                rewrite_state["messages"] = checkpoint["channel_values"].get(
                    "messages", []
                )

            rewrite_result = rewrite_query(rewrite_state)
            search_question = rewrite_result.get("question", req.question)

            # ── Stage 2: Retrieve candidates from Qdrant ─────────────────
            search_kwargs: dict = {"k": RETRIEVAL_K}
            if SIMILARITY_THRESHOLD > 0.0:
                search_kwargs["score_threshold"] = SIMILARITY_THRESHOLD
            if HYBRID_FUSION_ALPHA != 0.5:
                search_kwargs["alpha"] = HYBRID_FUSION_ALPHA

            # Filter to current document only
            if qdrant_source:
                from qdrant_client.models import FieldCondition, MatchValue, Filter
                search_kwargs["filter"] = Filter(must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value=qdrant_source))
                ])

            base_retriever = get_store().as_retriever(
                search_type="similarity", search_kwargs=search_kwargs,
            )
            candidates = base_retriever.invoke(search_question)

            # ── Stage 3: Rerank candidates ───────────────────────────────
            reranker = _build_reranker()
            context_docs = list(
                reranker.compress_documents(candidates, search_question)
            )
            context_docs = _filter_by_score(context_docs, RERANK_SCORE_THRESHOLD)

            source_pages = sorted(
                set(
                    d.metadata.get("page", 0)
                    for d in context_docs
                    if d.metadata.get("page")
                )
            )

            # ── Stage 4: Stream LLM answer token by token ───────────────
            context_str = _format_docs(context_docs)
            checkpoint = checkpointer.get(config)
            history: list = []
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

            # ── Stage 5: Persist conversation in checkpointer ──────────
            from langchain_core.messages import AIMessage

            try:
                save_config = {**config, "checkpoint_ns": "", "checkpoint_id": None}
                checkpoint = checkpointer.get(save_config)
                prev_msgs = []
                if checkpoint and "channel_values" in checkpoint:
                    prev_msgs = checkpoint["channel_values"].get("messages", [])
                new_msgs = prev_msgs + [
                    HumanMessage(content=req.question),
                    AIMessage(content=full_answer),
                ]
                checkpointer.put(
                    save_config,
                    {"channel_values": {"messages": new_msgs}},
                    {"source": "input", "step": len(new_msgs), "writes": {}},
                    {},
                )
            except Exception as save_err:
                log.warning("Failed to save conversation: %s", save_err)

            # Emit source pages and final done event
            if source_pages:
                yield f"data: {json.dumps({'type': 'sources', 'pages': source_pages})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'answer': full_answer})}\n\n"

        except Exception as e:
            log.error("Chat error (session=%s): %s", req.session_id, e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS,
    )
