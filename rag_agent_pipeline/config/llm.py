"""LLM configuration — one instance per pipeline step.

Each node that calls an LLM gets its own model via LLM_MODEL_<STEP>.
All steps share a common base URL and API key, but each can use a
different model (e.g. a lighter model for rewriting, a stronger one
for generation).
"""

# Module pour accéder aux variables d'environnement
import os

# Client LangChain compatible OpenAI pour les LLM
from langchain_openai import ChatOpenAI

# S'assure que le fichier .env est chargé avant de lire les variables
import config.env  # noqa: F401

# ── Paramètres partagés par tous les LLM ─────────────────────────────────────

# URL de base de l'API compatible OpenAI
LLM_BASE_URL    = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
# Clé API du fournisseur LLM
LLM_API_KEY     = os.getenv("LLM_API_KEY", "")
# Température de génération (0 = déterministe)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# ── Modèle par étape ─────────────────────────────────────────────────────────
# Chaque nœud du pipeline a sa propre variable LLM_MODEL_<STEP>.
# Cela permet d'utiliser un modèle léger pour les tâches simples
# (rewrite, expand, hyde, self_query) et un modèle plus puissant
# pour la génération de la réponse finale.

LLM_MODEL_REWRITE    = os.getenv("LLM_MODEL_REWRITE", "gemini-2.5-flash-lite")
LLM_MODEL_EXPAND     = os.getenv("LLM_MODEL_EXPAND", "gemini-2.5-flash-lite")
LLM_MODEL_HYDE       = os.getenv("LLM_MODEL_HYDE", "gemini-2.5-flash-lite")
LLM_MODEL_SELF_QUERY = os.getenv("LLM_MODEL_SELF_QUERY", "gemini-2.5-flash-lite")
LLM_MODEL_GENERATE   = os.getenv("LLM_MODEL_GENERATE", "gemini-2.5-flash")


def _build_llm(model: str, streaming: bool = False, max_tokens: int | None = None) -> ChatOpenAI:
    """Construit une instance ChatOpenAI avec les paramètres partagés.

    Args:
        model: identifiant du modèle (ex: gemini-2.5-flash-lite)
        streaming: activer le streaming token par token (seulement pour generate)
        max_tokens: limite de tokens en sortie (None = défaut du provider)
    """
    return ChatOpenAI(
        model=model,
        temperature=LLM_TEMPERATURE,
        max_tokens=max_tokens,
        streaming=streaming,
        openai_api_key=LLM_API_KEY,
        openai_api_base=LLM_BASE_URL,
    )


# ── Instances LLM par étape ──────────────────────────────────────────────────

# Réécriture de requête — modèle léger, pas de streaming, réponses courtes
LLM_REWRITE    = _build_llm(LLM_MODEL_REWRITE, max_tokens=256)
# Expansion de requête — modèle léger, pas de streaming
LLM_EXPAND     = _build_llm(LLM_MODEL_EXPAND, max_tokens=512)
# Génération de document hypothétique (HyDE) — modèle léger
LLM_HYDE       = _build_llm(LLM_MODEL_HYDE, max_tokens=512)
# Extraction de filtres de métadonnées — modèle léger, réponses très courtes
LLM_SELF_QUERY = _build_llm(LLM_MODEL_SELF_QUERY, max_tokens=128)
# Génération de la réponse finale — modèle principal avec streaming
LLM_GENERATE   = _build_llm(LLM_MODEL_GENERATE, streaming=True)

# Alias de rétro-compatibilité pour les imports existants (chat.py)
LLM = LLM_GENERATE
