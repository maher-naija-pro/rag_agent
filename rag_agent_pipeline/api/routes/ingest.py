"""POST /api/ingest — upload a PDF and run the ingestion pipeline.
GET  /api/ingest/{job_id} — poll ingestion status.
"""

# Active les annotations différées pour le typage
from __future__ import annotations

# Module de génération d'identifiants uniques UUID
import uuid

# Composants FastAPI pour le routage et l'upload
from fastapi import APIRouter, HTTPException, UploadFile, File
# Type de message utilisateur LangChain
from langchain_core.messages import HumanMessage

# État partagé et constantes de configuration
from api.state import graph, sessions, jobs, UPLOAD_DIR, MAX_FILE_SIZE
# Fonction utilitaire pour créer un logger
from logger import get_logger

# Crée un routeur pour les routes d'ingestion
router = APIRouter()
# Crée un logger nommé pour ce module
log = get_logger("api.routes.ingest")


# Déclare une route POST avec code de retour 202 (accepté)
@router.post("/api/ingest", status_code=202)
# Reçoit le fichier uploadé en paramètre
async def ingest(file: UploadFile = File(...)):
    # Vérifie que le fichier est un PDF par son extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        # Erreur 400 si le fichier n'est pas un PDF
        raise HTTPException(400, "Only PDF files are accepted")

    # Vérifie le type MIME du fichier
    if file.content_type and file.content_type != "application/pdf":
        # Erreur 400 si le type MIME est incorrect
        raise HTTPException(400, "Only PDF files are accepted")

    # Lit le contenu binaire du fichier uploadé
    content = await file.read()
    # Vérifie que le fichier ne dépasse pas la taille maximale
    if len(content) > MAX_FILE_SIZE:
        # Erreur 413 si fichier trop volumineux
        raise HTTPException(413, f"File too large. Max: {MAX_FILE_SIZE // 1024 // 1024} MB")

    # Génère un identifiant unique pour la tâche d'ingestion
    job_id     = str(uuid.uuid4())
    # Génère un identifiant unique pour la session
    session_id = str(uuid.uuid4())
    # Génère un identifiant unique pour le fil de conversation
    thread_id  = str(uuid.uuid4())
    # Génère un identifiant unique pour le document
    doc_id     = str(uuid.uuid4())
    # Construit le chemin de sauvegarde du fichier
    file_path  = UPLOAD_DIR / f"{doc_id}.pdf"
    # Écrit le contenu du fichier sur le disque
    file_path.write_bytes(content)

    # Enregistre les métadonnées de la tâche d'ingestion
    jobs[job_id] = {
        # Statut initial : en cours de traitement
        "status": "processing",
        # Progression initiale à 0%
        "progress": 0,
        # Identifiant de session associé
        "session_id": session_id,
        # Identifiant du document associé
        "document_id": doc_id,
        # Nom original du fichier
        "file_name": file.filename,
        # Chemin du fichier sauvegardé
        "file_path": str(file_path),
    }

    # Journalise le début de l'ingestion
    log.info("Ingesting '%s' (job=%s, session=%s)", file.filename, job_id, session_id)

    try:
        # Met à jour la progression à 10%
        jobs[job_id]["progress"] = 10

        # Exécute le graphe d'ingestion du pipeline RAG
        result = graph.invoke(
            {
                # Message initial pour le graphe
                "messages":  [HumanMessage(content="Ingesting document")],
                # Chemin du fichier à ingérer
                "question":  str(file_path),
                # Source vide, sera déterminée par le pipeline
                "source":    "",
                # Pages brutes extraites (initialement vide)
                "raw_pages": [],
                # Morceaux de texte (initialement vide)
                "chunks":    [],
                # Documents de contexte (initialement vide)
                "context":   [],
                # Réponse (non utilisé pour l'ingestion)
                "answer":    "",
                # Indicateur d'ingestion (initialement faux)
                "ingested":  False,
            },
            # Configuration avec l'identifiant de fil
            config={"configurable": {"thread_id": thread_id}},
        )

        # Compte le nombre de pages extraites
        pages_count  = len(result.get("raw_pages", []))
        # Compte le nombre de morceaux de texte créés
        chunks_count = len(result.get("chunks", []))

        # Enregistre la session avec les métadonnées du document ingéré
        sessions[session_id] = {
            # Identifiant du fil de conversation
            "thread_id":   thread_id,
            # Identifiant du document
            "document_id": doc_id,
            # Nom original du fichier
            "file_name":   file.filename,
            # Chemin du fichier sauvegardé
            "file_path":   str(file_path),
            # Nombre de pages extraites
            "pages":       pages_count,
            # Nombre de morceaux indexés
            "chunks":      chunks_count,
        }

        # Met à jour le statut de la tâche comme terminée
        jobs[job_id].update({
            # Statut prêt et progression à 100%
            "status": "ready", "progress": 100,
            # Ajoute les compteurs de pages et morceaux
            "pages": pages_count, "chunks": chunks_count,
        })

    # Capture toute erreur survenue pendant l'ingestion
    except Exception as e:
        # Journalise l'erreur d'ingestion
        log.error("Ingestion failed for job %s: %s", job_id, e)
        # Met à jour le statut comme échoué
        jobs[job_id].update({"status": "failed", "error": str(e)})
        # Retourne une erreur 500 au client
        raise HTTPException(500, f"Ingestion failed: {e}")

    # Journalise la fin de l'ingestion
    log.info("Ingestion complete: %d pages, %d chunks (job=%s)", pages_count, chunks_count, job_id)

    # Retourne le résultat de l'ingestion au client
    return {
        # Identifiants générés
        "job_id": job_id, "session_id": session_id, "document_id": doc_id,
        # Informations sur le document
        "file_name": file.filename, "pages": pages_count, "chunks": chunks_count,
        # Statut final
        "status": "ready",
    }


# Déclare une route GET pour consulter le statut d'une tâche
@router.get("/api/ingest/{job_id}")
# Reçoit l'identifiant de la tâche en paramètre
async def ingest_status(job_id: str):
    # Recherche la tâche dans le dictionnaire
    job = jobs.get(job_id)
    # Vérifie que la tâche existe
    if not job:
        # Erreur 404 si tâche introuvable
        raise HTTPException(404, "Job not found")

    # Journalise la consultation du statut
    log.debug("Job status lookup (job=%s, status=%s)", job_id, job.get("status"))
    # Retourne les détails de la tâche au client
    return {
        # Identifiant de la tâche
        "job_id": job_id,
        # Statut actuel de la tâche
        "status":      job["status"],
        # Pourcentage de progression
        "progress":    job.get("progress", 0),
        # Identifiant de session associé
        "session_id":  job.get("session_id"),
        # Identifiant du document associé
        "document_id": job.get("document_id"),
        # Nom original du fichier
        "file_name":   job.get("file_name"),
        # Nombre de pages extraites
        "pages":       job.get("pages", 0),
        # Nombre de morceaux créés
        "chunks":      job.get("chunks", 0),
        # Message d'erreur éventuel
        "error":       job.get("error"),
    }
