# Modèles Pydantic pour les requêtes et réponses de l'API
"""Pydantic request / response models for the RAG API."""

# Importe la classe de base pour la validation de données
from pydantic import BaseModel


# Modèle de requête pour l'endpoint de chat
class ChatRequest(BaseModel):
    """Body for POST /api/chat."""

    # Identifiant de session retourné par POST /api/ingest
    session_id: str
    # Question en langage naturel posée par l'utilisateur
    question: str


# Modèle d'un événement SSE envoyé pendant une réponse de chat
class ChatEvent(BaseModel):
    """Single Server-Sent Event emitted during a chat response.

    Event types:
        token   — incremental answer text (content field)
        sources — page numbers referenced (pages field)
        done    — full final answer (answer field)
        error   — error message (content field)
    """

    # Type d'événement : "token", "sources", "done" ou "error"
    type: str
    # Texte du token ou message d'erreur
    content: str = ""
    # Numéros de pages sources (uniquement pour l'événement sources)
    pages: list[int] = []
    # Texte complet de la réponse (uniquement pour l'événement done)
    answer: str = ""


# Modèle de métadonnées d'un document ingéré
class DocumentInfo(BaseModel):
    """Document metadata returned by GET /api/documents."""

    # Identifiant unique UUID du document
    id: str
    # Nom original du fichier uploadé
    name: str
    # Nombre de pages extraites du PDF
    pages: int
    # Nombre de morceaux de texte stockés dans Qdrant
    chunks: int
    # Statut du document : "ready", "processing" ou "failed"
    status: str
    # Identifiant de session à utiliser pour les requêtes de chat
    session_id: str
