"""RAG pipeline state definition."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    messages:   Annotated[list[BaseMessage], add_messages]
    question:   str                  # latest user question (may be rewritten)
    original_question: str           # original user question before rewriting
    expanded_queries: list[str]      # variant queries from query expansion
    metadata_filter: dict            # extracted metadata filters from self-query (e.g. {"page": 5, "language": "fr"})
    source:     str                  # PDF filename — used to filter retrieval to this document
    raw_pages:  list[Document]       # pages from PDF loader
    chunks:     list[Document]       # chunks from text splitter
    candidates: list[Document]       # raw retrieval results (before reranking)
    context:    list[Document]       # reranked retrieval results
    answer:     str                  # final LLM answer
    cache_hit:  bool                 # True if answer was served from semantic cache
    ingested:   bool                 # True once the PDF has been embedded
