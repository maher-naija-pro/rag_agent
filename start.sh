#!/usr/bin/env bash
# Start the full RAG stack: Docker, API, and UI.
# Usage: ./start.sh [--install]
#   --install   Install Python and Node dependencies before starting

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL=false

for arg in "$@"; do
    case "$arg" in
        --install) INSTALL=true ;;
        *) echo "Unknown option: $arg"; echo "Usage: ./start.sh [--install]"; exit 1 ;;
    esac
done

cleanup() {
    echo "Shutting down..."
    kill "$API_PID" "$UI_PID" 2>/dev/null
    wait "$API_PID" "$UI_PID" 2>/dev/null
    docker compose -f "$ROOT_DIR/docker-compose.yml" down
    echo "Stopped."
}
trap cleanup EXIT INT TERM

if [ "$INSTALL" = true ]; then
    echo "Installing Python dependencies..."
    pip install -r "$ROOT_DIR/rag_agent_pipeline/requirements.txt"

    echo "Installing Node dependencies..."
    cd "$ROOT_DIR/UI" && npm install
    cd "$ROOT_DIR"
fi

# Start infrastructure
echo "Starting Docker services (Qdrant, Postgres)..."
docker compose -f "$ROOT_DIR/docker-compose.yml" up -d

# Start API
echo "Starting API server..."
cd "$ROOT_DIR/rag_agent_pipeline"
python main.py &
API_PID=$!

# Start UI
echo "Starting UI dev server..."
cd "$ROOT_DIR/UI"
npm run dev &
UI_PID=$!

echo "API: http://localhost:8000"
echo "UI:  http://localhost:3000"
echo "Press Ctrl+C to stop both."

wait
