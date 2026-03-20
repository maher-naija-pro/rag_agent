"""LLM configuration — Ollama via OpenAI-compatible API."""

# Module pour accéder aux variables d'environnement
import os

# Client LangChain compatible OpenAI pour les LLM
from langchain_openai import ChatOpenAI

# S'assure que le fichier .env est chargé avant de lire les variables
import config.env  # noqa: F401

# URL de base du serveur Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# Nom du modèle LLM à utiliser
LLM_MODEL       = os.getenv("LLM_MODEL", "mistral:7b")
# Température de génération (0 = déterministe)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
# Nombre max de tokens (0 = valeur par défaut du fournisseur)
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "0")) or None

# Crée l'instance singleton du LLM via l'API compatible OpenAI
LLM = ChatOpenAI(
    # Modèle à utiliser
    model=LLM_MODEL,
    # Température de génération
    temperature=LLM_TEMPERATURE,
    # Limite de tokens en sortie
    max_tokens=LLM_MAX_TOKENS,
    # Active le streaming des réponses token par token
    streaming=True,
    # Clé API factice car Ollama n'en nécessite pas
    openai_api_key="ollama",
    # Pointe vers le serveur Ollama local
    openai_api_base=OLLAMA_BASE_URL,
)
