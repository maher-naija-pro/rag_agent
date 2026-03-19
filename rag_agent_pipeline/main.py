"""Entry point — start the RAG API server.

Usage:
    python main.py
    python main.py --port 8080
    python main.py --host 127.0.0.1 --port 8080 --reload
"""

import argparse

import uvicorn

from config import API_HOST, API_PORT
from logger import get_logger

log = get_logger("main")


def main() -> None:
    """Parse CLI arguments and start the Uvicorn server."""
    parser = argparse.ArgumentParser(description="RAG Pipeline API server")
    parser.add_argument("--host", default=API_HOST, help=f"Bind address (default: {API_HOST})")
    parser.add_argument("--port", type=int, default=API_PORT, help=f"Port (default: {API_PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    log.info("Starting server on %s:%d (reload=%s)", args.host, args.port, args.reload)
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
