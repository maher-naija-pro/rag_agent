#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OLLAMA_MODEL="${OLLAMA_MODEL:-mistral:7b}"

# ── Check system dependencies ────────────────────────────────────────────────
if ! command -v tesseract &>/dev/null; then
    echo "ERROR: tesseract-ocr is required. Install with:"
    echo "  sudo apt-get install -y tesseract-ocr tesseract-ocr-fra"
    exit 1
fi

# ── Start test services ──────────────────────────────────────────────────────
echo "▶ Starting test services (Qdrant + Ollama)…"
docker compose -f docker-compose.test.yml up -d --wait

# ── Pull the LLM model if not already present ────────────────────────────────
echo "▶ Ensuring Ollama model '${OLLAMA_MODEL}' is available…"
docker compose -f docker-compose.test.yml exec ollama-test ollama pull "${OLLAMA_MODEL}"

# ── Run tests ─────────────────────────────────────────────────────────────────
echo "▶ Running pytest…"
cd rag_agent_pipeline
OLLAMA_BASE_URL="http://localhost:11434/v1" \
LLM_MODEL="${OLLAMA_MODEL}" \
QDRANT_URL="http://localhost:6333" \
python -m pytest "${@:---tb=short -q}"
EXIT_CODE=$?

# ── Tear down ─────────────────────────────────────────────────────────────────
cd ..
echo "▶ Stopping test services…"
docker compose -f docker-compose.test.yml down

exit $EXIT_CODE
