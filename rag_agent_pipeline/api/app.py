"""FastAPI application — assembles routers and middleware.

Usage:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

# Active les annotations différées pour le typage
from __future__ import annotations

# Importe le framework web FastAPI
from fastapi import FastAPI
# Middleware pour gérer les requêtes cross-origin
from fastapi.middleware.cors import CORSMiddleware

# Importe les routeurs de chaque ressource
from api.routes import health, ingest, chat, documents
# Origines autorisées pour les requêtes CORS
from config import CORS_ORIGINS
# Fonction utilitaire pour créer un logger
from logger import get_logger

# Crée un logger nommé pour ce module
log = get_logger("api.app")

# Instancie l'application FastAPI avec ses métadonnées
app = FastAPI(
    # Titre de l'API affiché dans la documentation
    title="RAG Pipeline API",
    # Version de l'API
    version="1.0.0",
    # Description de l'API
    description="PDF ingestion and AI-powered document Q&A",
)

# CORS — allow the frontend to call us
# Ajoute le middleware CORS à l'application
app.add_middleware(
    CORSMiddleware,
    # Liste des origines autorisées
    allow_origins=CORS_ORIGINS,
    # Autorise l'envoi de cookies et identifiants
    allow_credentials=True,
    # Autorise toutes les méthodes HTTP
    allow_methods=["*"],
    # Autorise tous les en-têtes HTTP
    allow_headers=["*"],
)

# Register route groups
# Enregistre les routes de vérification de santé
app.include_router(health.router)
# Enregistre les routes d'ingestion de documents
app.include_router(ingest.router)
# Enregistre les routes de chat / questions-réponses
app.include_router(chat.router)
# Enregistre les routes de gestion des documents
app.include_router(documents.router)

# Journalise l'initialisation de l'application
log.info("FastAPI app initialized (CORS origins: %s)", CORS_ORIGINS)
