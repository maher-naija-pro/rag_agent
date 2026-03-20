# Endpoint de vérification de santé et connectivité Qdrant
"""GET /api/health — liveness + Qdrant connectivity check."""

# Importe le routeur pour regrouper les routes
from fastapi import APIRouter

# Importe le dictionnaire des sessions actives
from api.state import sessions
# Fonction utilitaire pour créer un logger
from logger import get_logger

# Crée un routeur pour les routes de santé
router = APIRouter()
# Crée un logger nommé pour ce module
log = get_logger("api.routes.health")


# Déclare une route GET sur /api/health
@router.get("/api/health")
async def health():
    # Importe le client Qdrant à la demande (import paresseux)
    from config import get_client

    try:
        # Récupère une instance du client Qdrant
        client = get_client()
        # Liste toutes les collections Qdrant
        collections = client.get_collections()
        # Retourne un statut sain avec les informations de connexion
        return {
            "status": "healthy",
            "qdrant": "connected",
            # Nombre de collections trouvées
            "collections": len(collections.collections),
            # Nombre de sessions actives
            "sessions": len(sessions),
        }
    # Capture toute erreur de connexion à Qdrant
    except Exception as e:
        # Journalise l'avertissement
        log.warning("Health check: Qdrant unreachable — %s", e)
        # Retourne un statut dégradé avec le détail de l'erreur
        return {
            "status": "degraded",
            "qdrant": f"error: {e}",
            # Nombre de sessions actives malgré l'erreur
            "sessions": len(sessions),
        }
