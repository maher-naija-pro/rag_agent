"""Pydantic request / response models for the RAG API."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Body for POST /api/chat and POST /api/chat/stream."""

    session_id: str   # returned by POST /api/ingest
    question: str     # user's natural-language question


class ChatEvent(BaseModel):
    """Single Server-Sent Event emitted during a chat response.

    Event types:
        token   — incremental answer text (content field)
        sources — page numbers referenced (pages field)
        done    — full final answer (answer field)
        error   — error message (content field)
    """

    type: str           # "token" | "sources" | "done" | "error"
    content: str = ""   # token text or error message
    pages: list[int] = []  # source page numbers (sources event only)
    answer: str = ""    # complete answer text (done event only)


class DocumentInfo(BaseModel):
    """Document metadata returned by GET /api/documents."""

    id: str         # unique document UUID
    name: str       # original filename
    pages: int      # number of pages extracted from the PDF
    chunks: int     # number of text chunks stored in Qdrant
    status: str     # "ready" | "processing" | "failed"
    session_id: str # session to use for chat requests
