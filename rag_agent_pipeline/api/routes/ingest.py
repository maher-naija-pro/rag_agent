"""POST /api/ingest — upload a PDF and run the ingestion pipeline.
GET  /api/ingest/{job_id} — poll ingestion status.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File
from langchain_core.messages import HumanMessage

from api.state import graph, sessions, jobs, UPLOAD_DIR, MAX_FILE_SIZE
from logger import get_logger

router = APIRouter()
log = get_logger("api.routes.ingest")


@router.post("/api/ingest", status_code=202)
async def ingest(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    if file.content_type and file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are accepted")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max: {MAX_FILE_SIZE // 1024 // 1024} MB")

    job_id     = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    thread_id  = str(uuid.uuid4())
    doc_id     = str(uuid.uuid4())
    file_path  = UPLOAD_DIR / f"{doc_id}.pdf"
    file_path.write_bytes(content)

    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "session_id": session_id,
        "document_id": doc_id,
        "file_name": file.filename,
        "file_path": str(file_path),
    }

    log.info("Ingesting '%s' (job=%s, session=%s)", file.filename, job_id, session_id)

    try:
        jobs[job_id]["progress"] = 10

        result = graph.invoke(
            {
                "messages":  [HumanMessage(content="Ingesting document")],
                "question":  str(file_path),
                "source":    "",
                "raw_pages": [],
                "chunks":    [],
                "context":   [],
                "answer":    "",
                "ingested":  False,
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        pages_count  = len(result.get("raw_pages", []))
        chunks_count = len(result.get("chunks", []))

        sessions[session_id] = {
            "thread_id":   thread_id,
            "document_id": doc_id,
            "file_name":   file.filename,
            "file_path":   str(file_path),
            "pages":       pages_count,
            "chunks":      chunks_count,
        }

        jobs[job_id].update({
            "status": "ready", "progress": 100,
            "pages": pages_count, "chunks": chunks_count,
        })

    except Exception as e:
        log.error("Ingestion failed for job %s: %s", job_id, e)
        jobs[job_id].update({"status": "failed", "error": str(e)})
        raise HTTPException(500, f"Ingestion failed: {e}")

    log.info("Ingestion complete: %d pages, %d chunks (job=%s)", pages_count, chunks_count, job_id)

    return {
        "job_id": job_id, "session_id": session_id, "document_id": doc_id,
        "file_name": file.filename, "pages": pages_count, "chunks": chunks_count,
        "status": "ready",
    }


@router.get("/api/ingest/{job_id}")
async def ingest_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    log.debug("Job status lookup (job=%s, status=%s)", job_id, job.get("status"))
    return {
        "job_id": job_id,
        "status":      job["status"],
        "progress":    job.get("progress", 0),
        "session_id":  job.get("session_id"),
        "document_id": job.get("document_id"),
        "file_name":   job.get("file_name"),
        "pages":       job.get("pages", 0),
        "chunks":      job.get("chunks", 0),
        "error":       job.get("error"),
    }
