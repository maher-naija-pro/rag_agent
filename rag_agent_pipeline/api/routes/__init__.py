# Sous-paquet des routes, un fichier par ressource
"""Route sub-package — one file per resource."""

# Importe tous les modules de routes
from api.routes import health, ingest, chat, documents

# Expose les modules de routes lors d'un import *
__all__ = ["health", "ingest", "chat", "documents"]
