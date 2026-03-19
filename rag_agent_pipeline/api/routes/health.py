"""GET /api/health — liveness + Qdrant connectivity check."""

from fastapi import APIRouter

from api.state import sessions
from logger import get_logger

router = APIRouter()
log = get_logger("api.routes.health")


@router.get("/api/health")
async def health():
    from config import get_client

    try:
        client = get_client()
        collections = client.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "collections": len(collections.collections),
            "sessions": len(sessions),
        }
    except Exception as e:
        log.warning("Health check: Qdrant unreachable — %s", e)
        return {
            "status": "degraded",
            "qdrant": f"error: {e}",
            "sessions": len(sessions),
        }
