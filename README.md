# RAG Agent Pipeline

A PDF question-answering system. You upload a document, ask questions about it, and get answers with page citations. Built on LangGraph, FastAPI, and Next.js.

## What's inside

- **Backend** — LangGraph pipeline with 13 nodes, served by FastAPI/Uvicorn
- **Frontend** — Next.js 16, shadcn/ui components
- **Search** — Qdrant for hybrid retrieval (dense embeddings + BM25 sparse)
- **LLM** — works with any OpenAI-compatible API (Gemini, Groq, OpenAI, etc.)
- **Eval** — RAGAS metrics for measuring answer quality

Under the hood the pipeline does: PDF loading with OCR fallback, text chunking, vector embedding, optional query rewriting, hybrid search, cross-encoder reranking, and streamed answer generation.

## Quick start

```bash
./start.sh              # boots Docker (Qdrant), API, and UI
./start.sh --install    # install deps first, then start
```

API at `localhost:8000`, UI at `localhost:3000`. Hit `Ctrl+C` to stop.

## Project layout

```
rag_agent_pipeline/       # Python backend
  main.py                 # entry point
  graph.py                # LangGraph pipeline wiring
  nodes/                  # one file per pipeline stage
  api/routes/             # REST endpoints
  config/                 # all env-driven settings
  tests/                  # pytest suite (209 tests, 98% coverage)
evaluation/               # RAGAS eval framework
UI/                       # Next.js frontend (3-panel PDF viewer + chat)
docker-compose.yml        # Qdrant
```

## API endpoints

- `POST /api/ingest` — upload a PDF for processing
- `GET  /api/ingest/{job_id}` — poll ingestion status
- `GET  /api/documents` — list uploaded docs
- `DELETE /api/documents/{id}` — remove a doc
- `POST /api/chat` — ask a question (streams tokens via SSE)
- `GET  /api/health` — readiness check

## Configuration

All settings live in `rag_agent_pipeline/.env`.

### LLM

Every pipeline step that calls an LLM has its own model variable, so you can run a cheap/fast model for preprocessing and a stronger one for the final answer.

They all share the same API endpoint and key:

```
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
LLM_API_KEY=your-key-here
LLM_TEMPERATURE=0
```

Per-step models:

```
LLM_MODEL_REWRITE=gemini-2.5-flash-lite      # reformulates the user question
LLM_MODEL_EXPAND=gemini-2.5-flash-lite       # generates query variants
LLM_MODEL_HYDE=gemini-2.5-flash-lite         # hypothetical document passage
LLM_MODEL_SELF_QUERY=gemini-2.5-flash-lite   # extracts page/language filters
LLM_MODEL_GENERATE=gemini-2.5-flash          # produces the final answer
```

### Retrieval & reranking

- `RETRIEVAL_K` — candidates fetched from Qdrant (default 10)
- `HYBRID_FUSION_ALPHA` — balance between dense and sparse search (0.5)
- `RERANK_ENABLED` / `RERANK_PROVIDER` — toggle + backend (FlashRank by default)

### Optional pipeline stages

Each can be turned on/off independently:

- `QUERY_REWRITE_ENABLED` — LLM rewrites vague questions
- `QUERY_EXPANSION_ENABLED` — generates multiple search queries
- `HYDE_ENABLED` — hypothetical document embedding
- `SELF_QUERY_ENABLED` — auto-extracts metadata filters
- `CACHE_ENABLED` — semantic cache for repeated questions

### Other

- `CHUNK_SIZE` / `CHUNK_OVERLAP` — text splitting (800 / 150)
- `OCR_LANGUAGES` — Tesseract language packs (`fra+eng`)

## Evaluation

```bash
./evaluate.sh                # run eval suite
./evaluate.sh --install      # install deps first
./evaluate.sh --pdf doc.pdf  # eval on a specific document
```

Details in `evaluation/METRICS.md`.

## Tests

```bash
./run-tests.sh          # starts Qdrant, runs pytest, cleans up
./run-tests.sh -v       # verbose
```

Needs `tesseract-ocr` installed locally. Docker services are handled automatically.

The Dockerfile accepts a `REQUIREMENTS` build arg (`requirements.txt` for prod, `requirements-test.txt` for tests).
