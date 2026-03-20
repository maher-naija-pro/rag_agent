"""Node — expand_query: generate variant queries for broader retrieval."""

from __future__ import annotations

# Importation des types de messages LangChain
from langchain_core.messages import HumanMessage, SystemMessage

# Importation du modèle de langage dédié à l'expansion de requête
from config import LLM_EXPAND as LLM
# Paramètres d'activation et de nombre de variantes
from config.pipeline import QUERY_EXPANSION_ENABLED, QUERY_EXPANSION_COUNT
# Importation du logger et de la fonction de représentation détaillée
from logger import get_logger, deep_repr
# Importation du type d'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.query_expander")

# Prompt système pour générer des variantes de la requête de recherche
EXPAND_PROMPT = """\
You are a search query generator for a document retrieval system.
Given a question, generate {count} alternative search queries that cover different angles, \
synonyms, or phrasings to find relevant document passages.

Rules:
- Keep the same language as the original question.
- Each variant should use different keywords or phrasing.
- Do NOT answer the question — only generate search queries.
- Output one query per line, no numbering, no bullet points.
"""


def expand_query(state: RAGState) -> dict:
    """
    Generate variant queries for broader retrieval coverage.

    - Takes the (already rewritten) question
    - Generates QUERY_EXPANSION_COUNT variants via LLM
    - Stores [original + variants] in expanded_queries
    - Skipped when QUERY_EXPANSION_ENABLED=false
    """
    # Journalisation de l'état d'entrée pour le débogage
    log.debug("[INPUT] expand_query:\n%s", deep_repr(dict(state)))
    # Récupération de la question (potentiellement déjà réécrite)
    question = state["question"]

    # Vérification si l'expansion de requête est désactivée
    if not QUERY_EXPANSION_ENABLED:
        log.debug("Query expansion disabled — single query")
        # Retour de la question seule sans variantes
        return {"expanded_queries": [question]}

    # Construction du prompt pour le LLM
    prompt = [
        # Instruction système avec le nombre de variantes demandé
        SystemMessage(content=EXPAND_PROMPT.format(count=QUERY_EXPANSION_COUNT)),
        # Question de l'utilisateur à développer en variantes
        HumanMessage(content=question),
    ]

    try:
        # Appel au LLM et nettoyage de la réponse
        response = LLM.invoke(prompt).content.strip()
        # Découpage de la réponse en lignes non vides
        variants = [line.strip() for line in response.split("\n") if line.strip()]

        # Filtrage des variantes trop courtes (moins de 5 caractères)
        variants = [v for v in variants if len(v) >= 5]

        # Limitation au nombre de variantes configuré
        variants = variants[:QUERY_EXPANSION_COUNT]

        # Inclusion systématique de la requête originale en première position
        # Ajout des variantes en évitant les doublons avec l'originale
        all_queries = [question] + [v for v in variants if v != question]

        # Journalisation du nombre de variantes générées
        log.info("Expanded '%s' → %d queries", question[:40], len(all_queries))
        # Construction du résultat avec toutes les requêtes
        result = {"expanded_queries": all_queries}
        # Journalisation du résultat de sortie
        log.debug("[OUTPUT] expand_query:\n%s", deep_repr(result))
        return result

    # Capture de toute exception lors de l'expansion
    except Exception as e:
        # Journalisation de l'erreur
        log.warning("Query expansion failed: %s — using single query", e)
        # Retour de la question seule en cas d'erreur
        result = {"expanded_queries": [question]}
        log.debug("[OUTPUT] expand_query:\n%s", deep_repr(result))
        return result
