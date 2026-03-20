"""Server, CORS, upload, and logging parameters."""

# Module pour accéder aux variables d'environnement
import os
# Module pour manipuler les chemins de fichiers
from pathlib import Path

# S'assure que le fichier .env est chargé
import config.env  # noqa: F401

# Serveur API
# Adresse d'écoute du serveur (0.0.0.0 = toutes les interfaces)
API_HOST = os.getenv("API_HOST", "0.0.0.0")
# Port d'écoute du serveur API
API_PORT = int(os.getenv("API_PORT", "8000"))

# CORS (Cross-Origin Resource Sharing)
# Liste des origines autorisées, séparées par des virgules
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# Upload de fichiers
# Répertoire temporaire pour les fichiers uploadés
UPLOAD_DIR      = Path(os.getenv("UPLOAD_DIR", "/tmp/rag_uploads"))
# Taille maximale d'upload en mégaoctets
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
# Conversion de la taille maximale en octets
MAX_FILE_SIZE   = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# Journalisation
# Niveau de log (converti en majuscules)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
