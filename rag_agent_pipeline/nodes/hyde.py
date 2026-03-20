"""Node — hyde: generate a hypothetical document for better retrieval."""

from __future__ import annotations

# Importation des types de messages LangChain
from langchain_core.messages import HumanMessage, SystemMessage

# Importation du modèle de langage configuré
from config import LLM
# Drapeau pour activer/désactiver la génération hypothétique (HyDE)
from config.pipeline import HYDE_ENABLED
# Importation du logger et de la fonction de représentation détaillée
from logger import get_logger, deep_repr
# Importation du type d'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.hyde")

# Prompt système pour générer un passage de document hypothétique
HYDE_PROMPT = """\
You are a document passage generator.
Given a question, write a short paragraph (3-5 sentences) that would appear in a document \
answering this question. Write it as if you are quoting from an actual document.

Rules:
- Keep the same language as the question (French → French, English → English).
- Write in a formal, document-like style (not conversational).
- Do NOT say "the document says" — write as if you ARE the document.
- Output ONLY the paragraph, nothing else.
"""


def hyde(state: RAGState) -> dict:
    """
    Generate a hypothetical document passage and add it to expanded_queries.

    The fake answer's embedding is closer to real document chunks
    than the original question, improving retrieval for knowledge-heavy queries.

    - Skipped when HYDE_ENABLED=false (default)
    - Appends the hypothetical passage to existing expanded_queries
    - On failure, passes through without modification
    """
    # Journalisation de l'état d'entrée pour le débogage
    log.debug("[INPUT] hyde:\n%s", deep_repr(dict(state)))
    # Vérification si HyDE est désactivé dans la configuration
    if not HYDE_ENABLED:
        log.debug("HyDE disabled — passing through")
        log.debug("[OUTPUT] hyde:\n%s", deep_repr({}))
        # Retour vide, aucune modification de l'état
        return {}

    # Récupération de la question de l'utilisateur
    question = state["question"]
    # Récupération des requêtes existantes ou utilisation de la question par défaut
    existing_queries = state.get("expanded_queries", [question])

    # Construction du prompt pour le LLM
    prompt = [
        # Instruction système pour la génération de passage hypothétique
        SystemMessage(content=HYDE_PROMPT),
        # Question de l'utilisateur comme entrée
        HumanMessage(content=question),
    ]

    try:
        # Appel au LLM pour générer le passage hypothétique
        hypothetical = LLM.invoke(prompt).content.strip()

        # Vérification que le passage généré est suffisamment long
        if not hypothetical or len(hypothetical) < 10:
            log.warning("HyDE returned empty — skipping")
            log.debug("[OUTPUT] hyde:\n%s", deep_repr({}))
            # Retour vide si le passage est trop court ou vide
            return {}

        # Ajout du passage hypothétique aux requêtes existantes
        updated_queries = existing_queries + [hypothetical]

        # Journalisation de la taille du passage généré
        log.info("HyDE: generated %d-char hypothetical passage", len(hypothetical))
        # Construction du résultat avec les requêtes mises à jour
        result = {"expanded_queries": updated_queries}
        # Journalisation du résultat de sortie
        log.debug("[OUTPUT] hyde:\n%s", deep_repr(result))
        return result

    # Capture de toute exception lors de la génération
    except Exception as e:
        # Journalisation de l'erreur
        log.warning("HyDE failed: %s — skipping", e)
        log.debug("[OUTPUT] hyde:\n%s", deep_repr({}))
        # Retour vide en cas d'erreur, l'état reste inchangé
        return {}
