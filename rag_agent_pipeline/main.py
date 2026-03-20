"""Entry point — start the RAG API server.

Usage:
    python main.py
    python main.py --port 8080
    python main.py --host 127.0.0.1 --port 8080 --reload
"""

# Module pour analyser les arguments de la ligne de commande
import argparse

# Serveur ASGI pour exécuter l'application web
import uvicorn

# Importation de l'hôte et du port par défaut depuis la configuration
from config import API_HOST, API_PORT
# Importation de la fonction pour obtenir un logger configuré
from logger import get_logger

# Création d'un logger nommé "main" pour ce module
log = get_logger("main")


def main() -> None:
    """Parse CLI arguments and start the Uvicorn server."""
    # Création du parseur d'arguments CLI
    parser = argparse.ArgumentParser(description="RAG Pipeline API server")
    # Argument pour l'adresse d'écoute du serveur
    parser.add_argument("--host", default=API_HOST, help=f"Bind address (default: {API_HOST})")
    # Argument pour le port d'écoute du serveur
    parser.add_argument("--port", type=int, default=API_PORT, help=f"Port (default: {API_PORT})")
    # Argument pour activer le rechargement automatique en développement
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    # Analyse des arguments fournis en ligne de commande
    args = parser.parse_args()

    # Journalisation du démarrage du serveur avec les paramètres choisis
    log.info("Starting server on %s:%d (reload=%s)", args.host, args.port, args.reload)
    # Lancement du serveur Uvicorn avec l'application FastAPI
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)


# Vérifie si le script est exécuté directement (et non importé)
if __name__ == "__main__":
    # Appel de la fonction principale pour démarrer le serveur
    main()
