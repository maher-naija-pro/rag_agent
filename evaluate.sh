#!/usr/bin/env bash
# Run RAGAS evaluation against the RAG pipeline.
# Usage: ./evaluate.sh [--install] [-- extra args for evaluate.py]
#   --install   Install evaluation dependencies first
#   Extra args are forwarded to evaluate.py (e.g. --pdf /path/to/doc.pdf)

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL=false
EXTRA_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--install" ]; then
        INSTALL=true
    elif [ "$arg" = "--" ]; then
        continue
    else
        EXTRA_ARGS+=("$arg")
    fi
done

if [ "$INSTALL" = true ]; then
    echo "Installing evaluation dependencies..."
    pip install -r "$ROOT_DIR/evaluation/requirements.txt"
fi

echo "Running evaluation..."
cd "$ROOT_DIR"
python -m evaluation.evaluate "${EXTRA_ARGS[@]}"
