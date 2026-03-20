"""GET    /api/documents              — list all ingested documents.
DELETE /api/documents/{document_id} — remove a document and its session.
"""

# Module de manipulation de chemins de fichiers
from pathlib import Path

# Routeur FastAPI et gestion des erreurs HTTP
from fastapi import APIRouter, HTTPException

# Importe le dictionnaire des sessions actives
from api.state import sessions
# Fonction utilitaire pour créer un logger
from logger import get_logger

# Crée un routeur pour les routes de gestion des documents
router = APIRouter()
# Crée un logger nommé pour ce module
log = get_logger("api.routes.documents")


# Déclare une route GET pour lister les documents
@router.get("/api/documents")
# Endpoint pour récupérer la liste de tous les documents ingérés
async def list_documents():
    # Initialise la liste des documents à retourner
    docs = []
    # Parcourt toutes les sessions actives
    for session_id, session in sessions.items():
        # Ajoute les métadonnées du document à la liste
        docs.append({
            # Identifiant unique du document
            "id":         session["document_id"],
            # Nom original du fichier
            "name":       session["file_name"],
            # Nombre de pages extraites
            "pages":      session.get("pages", 0),
            # Nombre de morceaux indexés
            "chunks":     session.get("chunks", 0),
            # Statut du document (toujours prêt car en session)
            "status":     "ready",
            # Identifiant de session pour les requêtes de chat
            "session_id": session_id,
        })
    # Journalise le nombre de documents listés
    log.debug("Listed %d documents", len(docs))
    # Retourne la liste des documents au client
    return {"documents": docs}


# Déclare une route DELETE pour supprimer un document
@router.delete("/api/documents/{document_id}")
# Reçoit l'identifiant du document à supprimer
async def delete_document(document_id: str):
    # Initialise l'identifiant de session cible
    target_session_id = None
    # Parcourt les sessions pour trouver le document
    for session_id, session in sessions.items():
        # Vérifie si le document correspond
        if session["document_id"] == document_id:
            # Mémorise l'identifiant de session trouvé
            target_session_id = session_id
            # Arrête la recherche dès que le document est trouvé
            break

    # Vérifie que le document a été trouvé
    if not target_session_id:
        # Erreur 404 si document introuvable
        raise HTTPException(404, "Document not found")

    # Supprime la session du dictionnaire et la récupère
    session   = sessions.pop(target_session_id)
    # Construit le chemin vers le fichier sur le disque
    file_path = Path(session.get("file_path", ""))
    try:
        # Vérifie que le fichier existe sur le disque
        if file_path.exists():
            # Supprime le fichier du disque
            file_path.unlink()
    # Capture les erreurs de suppression de fichier
    except OSError as e:
        # Journalise l'avertissement sans interrompre
        log.warning("Could not delete file '%s': %s", file_path, e)

    # Journalise la suppression réussie
    log.info("Deleted document %s (session=%s)", document_id, target_session_id)
    # Retourne la confirmation de suppression au client
    return {"status": "deleted", "document_id": document_id}
