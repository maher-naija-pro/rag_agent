#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OLLAMA_MODEL="${OLLAMA_MODEL:-mistral:7b}"

# ── Start services and build test image ──────────────────────────────────────
echo "▶ Building test image and starting services…"
docker compose -f docker-compose.test.yml build test-runner
docker compose -f docker-compose.test.yml up -d qdrant-test ollama-test

# ── Wait for Ollama and pull model ───────────────────────────────────────────
echo "▶ Waiting for Ollama…"
until docker compose -f docker-compose.test.yml exec ollama-test ollama list &>/dev/null; do
    sleep 2
done

echo "▶ Ensuring model '${OLLAMA_MODEL}' is available…"
docker compose -f docker-compose.test.yml exec ollama-test ollama pull "${OLLAMA_MODEL}"

# ── Run tests ─────────────────────────────────────────────────────────────────
echo "▶ Running pytest…"
docker compose -f docker-compose.test.yml run --rm \
    test-runner \
    python -m pytest tests/ -v --tb=short "${@}"
EXIT_CODE=$?

# ── Tear down ─────────────────────────────────────────────────────────────────
echo "▶ Stopping test services…"
docker compose -f docker-compose.test.yml down

exit $EXIT_CODE
