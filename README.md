# RAG Agent Pipeline

PDF document Q&A system built with LangGraph, FastAPI, and Next.js. Upload a PDF, ask questions, get answers with source citations.

## What's inside

- **Backend**: 13-node LangGraph pipeline (FastAPI + Uvicorn)
- **Frontend**: Next.js 16 with shadcn/ui
- **Vector store**: Qdrant (hybrid dense + sparse search)
- **LLM**: Ollama (local) or OpenAI API
- **Eval**: RAGAS metrics

The pipeline handles: PDF loading (with OCR fallback), chunking, embedding, query rewriting, hybrid retrieval, reranking (FlashRank), and streaming generation.

## Getting started

```bash
./start.sh              # starts Docker, API, and UI
./start.sh --install    # install deps first, then start
```

API runs on `localhost:8000`, UI on `localhost:3000`. `Ctrl+C` stops everything.

## Project layout

```
rag_agent_pipeline/       # Python backend
  main.py                 # entry point
  graph.py                # LangGraph pipeline definition
  nodes/                  # pipeline stages (loader, chunker, retriever, etc.)
  api/routes/             # REST endpoints
  tests/                  # pytest suite
evaluation/               # RAGAS eval framework
UI/                       # Next.js frontend
docker-compose.yml        # Qdrant + Postgres
```

## API

- `POST /api/ingest` — upload a PDF
- `GET /api/ingest/{job_id}` — check ingestion status
- `GET /api/documents` — list docs
- `DELETE /api/documents/{id}` — remove a doc
- `POST /api/chat` — ask a question (SSE streaming)
- `POST /api/chat/stream` — token-level streaming
- `GET /api/health`

## Config

Everything is configured via env vars in `.env`. The main knobs:

- `LLM_MODEL` — model to use (default: `mistral:7b`)
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking params (800 / 150)
- `RETRIEVAL_K` — how many chunks to retrieve (15)
- `HYBRID_FUSION_ALPHA` — dense vs sparse balance (0.5)
- `RERANK_ENABLED` / `RERANK_PROVIDER` — reranking toggle + backend
- `CACHE_ENABLED` — semantic cache for repeated questions
- `QUERY_REWRITE_ENABLED`, `HYDE_ENABLED` — query optimization toggles
- `OCR_LANGUAGES` — Tesseract languages (`fra+eng`)

## Evaluation

```bash
./evaluate.sh              # run eval
./evaluate.sh --install    # install deps first
./evaluate.sh --pdf doc.pdf  # eval on a specific PDF
```

Metrics docs in `evaluation/METRICS.md`.

## Tests

```bash
./run-tests.sh          # spins up Qdrant + Ollama, runs pytest, tears down
./run-tests.sh -v       # verbose output
```

Requires `tesseract-ocr` installed locally. The script handles Docker services automatically.
