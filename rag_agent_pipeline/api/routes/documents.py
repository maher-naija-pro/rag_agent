"""GET    /api/documents              — list all ingested documents.
DELETE /api/documents/{document_id} — remove a document and its session.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.state import sessions
from logger import get_logger

router = APIRouter()
log = get_logger("api.routes.documents")


@router.get("/api/documents")
async def list_documents():
    docs = []
    for session_id, session in sessions.items():
        docs.append({
            "id":         session["document_id"],
            "name":       session["file_name"],
            "pages":      session.get("pages", 0),
            "chunks":     session.get("chunks", 0),
            "status":     "ready",
            "session_id": session_id,
        })
    log.debug("Listed %d documents", len(docs))
    return {"documents": docs}


@router.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    target_session_id = None
    for session_id, session in sessions.items():
        if session["document_id"] == document_id:
            target_session_id = session_id
            break

    if not target_session_id:
        raise HTTPException(404, "Document not found")

    session   = sessions.pop(target_session_id)
    file_path = Path(session.get("file_path", ""))
    try:
        if file_path.exists():
            file_path.unlink()
    except OSError as e:
        log.warning("Could not delete file '%s': %s", file_path, e)

    log.info("Deleted document %s (session=%s)", document_id, target_session_id)
    return {"status": "deleted", "document_id": document_id}
