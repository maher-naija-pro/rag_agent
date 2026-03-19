"""FastAPI application — assembles routers and middleware.

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, ingest, chat, documents
from config import CORS_ORIGINS
from logger import get_logger

log = get_logger("api.app")

app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
    description="PDF ingestion and AI-powered document Q&A",
)

# CORS — allow the frontend to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route groups
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(documents.router)

log.info("FastAPI app initialized (CORS origins: %s)", CORS_ORIGINS)
