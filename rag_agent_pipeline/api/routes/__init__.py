"""Route sub-package — one file per resource."""

from api.routes import health, ingest, chat, documents

__all__ = ["health", "ingest", "chat", "documents"]
