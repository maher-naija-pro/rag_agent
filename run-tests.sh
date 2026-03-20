#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ── Check system dependencies ────────────────────────────────────────────────
if ! command -v tesseract &>/dev/null; then
    echo "ERROR: tesseract-ocr is required. Install with:"
    echo "  sudo apt-get install -y tesseract-ocr tesseract-ocr-fra"
    exit 1
fi

# ── Start test services ──────────────────────────────────────────────────────
echo "▶ Starting test services (Qdrant)…"
docker compose -f docker-compose.test.yml up -d --wait

# ── Run tests inside the test-runner container ───────────────────────────────
echo "▶ Running pytest…"
docker compose -f docker-compose.test.yml run --rm --build test-runner \
  python -m pytest "${@:---tb=short -q}"
EXIT_CODE=$?

# ── Tear down ─────────────────────────────────────────────────────────────────
echo "▶ Stopping test services…"
docker compose -f docker-compose.test.yml down

exit $EXIT_CODE
