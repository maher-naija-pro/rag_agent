"""Graph definition — wire all nodes together."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

from logger import get_logger
from state import RAGState
from nodes import (
    load_pdf, chunk, extract_metadata, embed_and_store,
    cache_check, cache_store,
    rewrite_query, expand_query, hyde, self_query,
    retrieve, rerank, generate,
)

log = get_logger("graph")


def should_ingest(state: RAGState) -> Literal["load_pdf", "cache_check"]:
    """
    On the first turn the PDF has not been ingested yet → route to load_pdf.
    On subsequent turns the store is already populated  → skip to cache_check.
    """
    if not state.get("ingested", False):
        log.info("First turn → routing to load_pdf")
        return "load_pdf"
    log.debug("Subsequent turn → routing to cache_check")
    return "cache_check"


def after_cache_check(state: RAGState) -> Literal["generate", "rewrite_query"]:
    """
    If cache hit → skip to generate (answer is already populated).
    If cache miss → continue with query processing pipeline.
    """
    if state.get("cache_hit", False):
        log.info("Cache hit → skipping to generate")
        return "generate"
    return "rewrite_query"


def build_graph(checkpointer: InMemorySaver):
    """
    Graph topology:

        START
          │
          ▼
      [router] ──── first turn ────► load_pdf → chunk → extract_metadata → embed_and_store ─┐
          │                                                                                  │
          └──── subsequent turns ────────────────────────────────────────────────────────────┤
                                                                                             ▼
                                                                                        cache_check
                                                                                             │
                                                                              ┌── HIT ──────┤
                                                                              │              └── MISS ──┐
                                                                              │                          ▼
                                                                              │                    rewrite_query
                                                                              │                          │
                                                                              │                     expand_query
                                                                              │                          │
                                                                              │                        hyde
                                                                              │                          │
                                                                              │                     self_query
                                                                              │                          │
                                                                              │                      retrieve
                                                                              │                          │
                                                                              │                       rerank
                                                                              │                          │
                                                                              └────────────► generate
                                                                                                 │
                                                                                            cache_store
                                                                                                 │
                                                                                                END
    """
    graph = StateGraph(RAGState)

    graph.add_node("load_pdf",          load_pdf)
    graph.add_node("chunk",             chunk)
    graph.add_node("extract_metadata",  extract_metadata)
    graph.add_node("embed_and_store",   embed_and_store)
    graph.add_node("cache_check",       cache_check)
    graph.add_node("rewrite_query",     rewrite_query)
    graph.add_node("expand_query",      expand_query)
    graph.add_node("hyde",              hyde)
    graph.add_node("self_query",        self_query)
    graph.add_node("retrieve",          retrieve)
    graph.add_node("rerank",            rerank)
    graph.add_node("generate",          generate)
    graph.add_node("cache_store",       cache_store)

    # Entry router
    graph.add_conditional_edges(
        START,
        should_ingest,
        {"load_pdf": "load_pdf", "cache_check": "cache_check"},
    )

    # Ingestion pipeline
    graph.add_edge("load_pdf",          "chunk")
    graph.add_edge("chunk",             "extract_metadata")
    graph.add_edge("extract_metadata",  "embed_and_store")
    graph.add_edge("embed_and_store",   "cache_check")

    # Cache routing — hit skips to generate, miss continues pipeline
    graph.add_conditional_edges(
        "cache_check",
        after_cache_check,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )

    # Query processing pipeline
    graph.add_edge("rewrite_query",    "expand_query")
    graph.add_edge("expand_query",     "hyde")
    graph.add_edge("hyde",             "self_query")
    graph.add_edge("self_query",       "retrieve")
    graph.add_edge("retrieve",        "rerank")
    graph.add_edge("rerank",          "generate")

    # Post-generation
    graph.add_edge("generate",        "cache_store")
    graph.add_edge("cache_store",     END)

    compiled = graph.compile(checkpointer=checkpointer)
    log.info("Graph compiled (13 nodes)")
    return compiled
