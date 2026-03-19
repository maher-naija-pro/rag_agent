"""Server, CORS, upload, and logging parameters."""

import os
from pathlib import Path

import config.env  # noqa: F401  ensure .env is loaded

# API Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Upload
UPLOAD_DIR      = Path(os.getenv("UPLOAD_DIR", "/tmp/rag_uploads"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
MAX_FILE_SIZE   = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
