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

# ── Run tests inside the test-runner container ───────────────────────────────
echo "▶ Running pytest…"
docker compose -f docker-compose.test.yml run --rm --build test-runner \
  python -m pytest "${@:---tb=short -q}"
EXIT_CODE=$?

# ── Tear down ─────────────────────────────────────────────────────────────────
echo "▶ Stopping test services…"
docker compose -f docker-compose.test.yml down

exit $EXIT_CODE
