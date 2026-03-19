"""Node 2 — chunk: split raw pages into overlapping chunks."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from logger import get_logger
from state import RAGState

log = get_logger("nodes.chunker")


def chunk(state: RAGState) -> dict:
    """Split raw pages into overlapping chunks; preserve page metadata."""
    raw_pages = state["raw_pages"]

    if not raw_pages:
        log.warning("No raw pages to chunk")
        return {"chunks": []}

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_pages)
    except Exception as e:
        log.error("Chunking failed: %s", e)
        return {"chunks": []}

    log.info("%d pages → %d chunks", len(raw_pages), len(chunks))
    return {"chunks": chunks}
