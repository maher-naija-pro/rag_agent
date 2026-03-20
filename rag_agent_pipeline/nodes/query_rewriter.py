"""Node — rewrite_query: reformulate user question for better retrieval."""

from __future__ import annotations

# Importation des types de messages LangChain
from langchain_core.messages import HumanMessage, SystemMessage

# Importation du modèle de langage dédié à la réécriture de requête
from config import LLM_REWRITE as LLM
# Drapeau pour activer/désactiver la réécriture de requête
from config.pipeline import QUERY_REWRITE_ENABLED
# Importation du logger et de la fonction de représentation détaillée
from logger import get_logger, deep_repr
# Importation du type d'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.query_rewriter")

# Prompt système qui guide le LLM pour réécrire la question de l'utilisateur
REWRITE_PROMPT = """\
You are a query rewriter for a document search system.
Rewrite the user's question to be clear, specific, and optimized for semantic search.

CRITICAL RULES:
- ALWAYS reply in the SAME language as the user's question. If the question is in French, rewrite in French. NEVER translate.
- The documents being searched are PDFs (contracts, reports, meeting minutes, legal documents). Keep the rewrite relevant to document search.
- Expand vague references ("it", "that", "this") using conversation history if available.
- Add relevant synonyms or related terms when helpful.
- Do NOT answer the question — only rewrite it.
- Output ONLY the rewritten question, nothing else. No explanation, no parentheses, no comments.
"""


def rewrite_query(state: RAGState) -> dict:
    """
    Reformulate the user's question for better retrieval.

    - Saves original question in original_question
    - Replaces question with a clearer, search-optimized version
    - Skipped when QUERY_REWRITE_ENABLED=false
    """
    # Journalisation de l'état d'entrée pour le débogage
    log.debug("[INPUT] rewrite_query:\n%s", deep_repr(dict(state)))
    # Récupération de la question de l'utilisateur depuis l'état
    question = state["question"]

    # Vérification si la réécriture est désactivée dans la configuration
    if not QUERY_REWRITE_ENABLED:
        log.debug("Query rewriting disabled — passing through")
        # Retour de la question originale sans modification
        result = {"original_question": question}
        # Journalisation du résultat de sortie
        log.debug("[OUTPUT] rewrite_query:\n%s", deep_repr(result))
        return result

    # Construction du contexte à partir de l'historique de conversation
    # Récupération de l'historique des messages
    history = state.get("messages", [])
    # Initialisation du contexte d'historique vide
    history_context = ""
    # Vérification s'il y a plus d'un message dans l'historique
    if len(history) > 1:
        # Extraction des 4 derniers messages (2 tours de conversation max)
        recent = history[-4:]
        # Concaténation des messages récents en texte formaté
        history_context = "\n".join(
            # Formatage avec rôle et contenu tronqué à 200 caractères
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:200]}"
            for m in recent
        )

    # Initialisation du message utilisateur avec la question brute
    user_msg = question
    # Si un contexte d'historique existe, on l'ajoute au message
    if history_context:
        # Assemblage du contexte et de la question
        user_msg = f"Conversation context:\n{history_context}\n\nQuestion to rewrite:\n{question}"

    # Construction du prompt complet pour le LLM
    prompt = [
        # Message système avec les instructions de réécriture
        SystemMessage(content=REWRITE_PROMPT),
        # Message utilisateur contenant la question à réécrire
        HumanMessage(content=user_msg),
    ]

    try:
        # Appel au LLM et nettoyage de la réponse
        rewritten = LLM.invoke(prompt).content.strip()

        # Conservation uniquement de la première ligne (le LLM ajoute parfois des explications)
        rewritten = rewritten.split("\n")[0].strip()

        # Suppression des guillemets si le LLM a encadré la réécriture
        # Détection des guillemets doubles autour du texte
        if rewritten.startswith('"') and rewritten.endswith('"'):
            # Retrait des guillemets
            rewritten = rewritten[1:-1]

        # Protection : si le LLM retourne un résultat vide ou trop court, on garde l'original
        # Vérification de la validité de la réécriture
        if not rewritten or len(rewritten) < 3:
            log.warning("Rewrite returned empty — keeping original")
            # Retour de la question originale en cas de résultat invalide
            result = {"original_question": question}
            log.debug("[OUTPUT] rewrite_query:\n%s", deep_repr(result))
            return result

        # Journalisation de la transformation effectuée
        log.info("Rewrite: '%s' → '%s'", question[:50], rewritten[:50])
        result = {
            # Sauvegarde de la question originale
            "original_question": question,
            # Remplacement par la question réécrite et optimisée
            "question": rewritten,
        }
        # Journalisation du résultat de sortie
        log.debug("[OUTPUT] rewrite_query:\n%s", deep_repr(result))
        return result

    # Capture de toute exception lors de la réécriture
    except Exception as e:
        # Journalisation de l'erreur
        log.warning("Query rewrite failed: %s — keeping original", e)
        # Retour de la question originale en cas d'erreur
        result = {"original_question": question}
        log.debug("[OUTPUT] rewrite_query:\n%s", deep_repr(result))
        return result
