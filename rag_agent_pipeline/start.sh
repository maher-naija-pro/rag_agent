#!/usr/bin/env bash
# Start the RAG API server.
# Usage: ./start.sh

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$DIR/venv"
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

# Load .env from parent or local directory
for f in "$DIR/../.env" "$DIR/.env"; do
  [ -f "$f" ] && set -a && source "$f" && set +a
done

# Create venv if missing
if [ ! -d "$VENV" ]; then
  echo "Creating venv..."
  python3 -m venv "$VENV"
fi

# Install deps
"$VENV/bin/pip" install -q -r "$DIR/requirements.txt"

# Start
cd "$DIR"
exec "$VENV/bin/uvicorn" api.app:app --host "$HOST" --port "$PORT" --reload
