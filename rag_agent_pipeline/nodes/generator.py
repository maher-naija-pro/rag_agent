"""Node 5 — generate: stream an answer from the LLM."""

from __future__ import annotations

# Importation du module système pour l'écriture en flux sur stdout
import sys

# Importation du type Document de LangChain
from langchain_core.documents import Document
# Importation des types de messages LangChain
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Importation du modèle de langage dédié à la génération de réponses
from config import LLM_GENERATE as LLM
# Importation du logger et de la fonction de représentation détaillée
from logger import get_logger, deep_repr
# Importation du type d'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.generator")

# Template du prompt système qui contient le contexte documentaire pour la génération
SYSTEM_TEMPLATE = """\
You are a precise, concise assistant. Answer only from the document excerpts below.

Rules:
- Be SHORT and DIRECT. Answer the question in 1-3 sentences maximum.
- Reply in the SAME language as the user's question (French → French, English → English).
- Cite page and line numbers exactly as shown in the context, e.g. [page 3, ligne 12] or [page 3, lignes 5-8]. If no line info, use [page 3].
- If the answer is absent, say so briefly — never fabricate.
- Do NOT repeat or summarize the entire document. Only give what was asked.

CONTEXT:
{context}
"""

# Message de refus par défaut quand aucun contexte documentaire n'est trouvé
NO_CONTEXT_ANSWER = (
    "I could not find any relevant information in the document to answer this question. "
    "Try rephrasing your question, or check that the document covers this topic."
)


def _format_docs(docs: list[Document]) -> str:
    """Render a list of documents into a single context string with page + line citations."""
    parts: list[str] = []
    for d in docs:
        page = d.metadata.get("page", "?")
        line_start = d.metadata.get("line_start")
        line_end = d.metadata.get("line_end")
        if line_start and line_end and line_start != line_end:
            ref = f"[page {page}, lignes {line_start}-{line_end}]"
        elif line_start:
            ref = f"[page {page}, ligne {line_start}]"
        else:
            ref = f"[page {page}]"
        parts.append(f"{ref}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def generate(state: RAGState) -> dict:
    """
    Stream an answer from the LLM using:
      - System prompt  containing the reranked context
      - History        from state["messages"] (populated by InMemorySaver)
      - Latest question

    If no context documents were retrieved, returns a fixed refusal
    instead of calling the LLM (prevents hallucination).
    """
    # Journalisation de l'état d'entrée pour le débogage
    log.debug("[INPUT] generate:\n%s", deep_repr(dict(state)))
    # Récupération des documents de contexte depuis l'état
    context_docs = state.get("context", [])

    # Protection : pas de contexte → refus de répondre pour éviter les hallucinations
    # Vérification si la liste de contexte est vide
    if not context_docs:
        log.warning("No context documents — returning refusal")
        result = {
            # Réponse de refus prédéfinie
            "answer":   NO_CONTEXT_ANSWER,
            # Ajout du message de refus à l'historique
            "messages": [AIMessage(content=NO_CONTEXT_ANSWER)],
        }
        # Journalisation du résultat de sortie
        log.debug("[OUTPUT] generate:\n%s", deep_repr(result))
        return result

    # Formatage des documents en une seule chaîne de contexte avec citations
    context_str = _format_docs(context_docs)
    # Récupération de l'historique sans le dernier message (la question courante)
    history     = state["messages"][:-1]

    # Construction du prompt complet pour le LLM
    prompt = (
        # Message système avec le contexte documentaire injecté
        [SystemMessage(content=SYSTEM_TEMPLATE.format(context=context_str))]
        # Ajout de l'historique de conversation pour le suivi multi-tours
        + history
        # Ajout de la question courante de l'utilisateur
        + [HumanMessage(content=state["question"])]
    )

    # Journalisation du début du streaming de la réponse
    log.info("Streaming answer …")
    # Initialisation de la liste pour accumuler les tokens de la réponse
    parts: list[str] = []
    try:
        # Écriture du préfixe de la réponse sur la sortie standard
        sys.stdout.write("Assistant: ")
        # Itération sur chaque token généré en streaming par le LLM
        for chunk_token in LLM.stream(prompt):
            # Extraction du contenu textuel du token
            tok = chunk_token.content
            # Écriture du token sur la sortie standard en temps réel
            sys.stdout.write(tok)
            # Vidage du tampon pour affichage immédiat du token
            sys.stdout.flush()
            # Accumulation du token dans la liste des parties
            parts.append(tok)
        # Saut de ligne après la fin de la réponse complète
        sys.stdout.write("\n")
    # Capture de toute exception lors du streaming
    except Exception as e:
        # Saut de ligne pour la propreté de l'affichage en cas d'erreur
        sys.stdout.write("\n")
        # Journalisation de l'erreur de streaming
        log.error("LLM streaming failed: %s", e)
        # Message d'erreur pour l'utilisateur
        error_msg = "An error occurred while generating the answer. Please try again."
        result = {
            # Retour du message d'erreur comme réponse
            "answer":   error_msg,
            # Ajout de l'erreur à l'historique des messages
            "messages": [AIMessage(content=error_msg)],
        }
        log.debug("[OUTPUT] generate:\n%s", deep_repr(result))
        return result

    # Reconstitution de la réponse complète à partir des tokens accumulés
    answer = "".join(parts)
    # Journalisation de la taille de la réponse générée
    log.info("Answer generated (%d chars)", len(answer))

    result = {
        # Stockage de la réponse finale générée
        "answer":   answer,
        # Ajout de la réponse à l'historique des messages pour le suivi
        "messages": [AIMessage(content=answer)],
    }
    # Journalisation du résultat de sortie
    log.debug("[OUTPUT] generate:\n%s", deep_repr(result))
    return result
