"""Node 5 — rerank: re-score candidates and keep the best N."""

from __future__ import annotations

# Importation du type Document de LangChain
from langchain_core.documents import Document

from config import (
    # Clé API pour le fournisseur de reranking
    RERANK_API_KEY,
    # Drapeau pour activer/désactiver le reranking
    RERANK_ENABLED,
    # Nom du modèle de reranking à utiliser
    RERANK_MODEL,
    # Fournisseur de reranking (flashrank, cohere, jina)
    RERANK_PROVIDER,
    # Seuil de score minimum pour conserver un document
    RERANK_SCORE_THRESHOLD,
    # Nombre maximum de documents à conserver après reranking
    RERANK_TOP_N,
)
# Importation du logger et de la fonction de représentation détaillée
from logger import get_logger, deep_repr
# Importation du type d'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.reranker")


# Instance singleton du reranker — None tant qu'elle n'a pas été initialisée
_reranker_instance = None
# Fournisseur utilisé pour l'instance en cache (pour détecter un changement de config)
_reranker_provider = None


def _build_reranker(provider: str | None = None):
    """Build or return the cached singleton reranker based on RERANK_PROVIDER.

    Le modèle de reranking (surtout FlashRank) est coûteux à charger en mémoire.
    On le construit une seule fois et on le réutilise pour toutes les requêtes.
    Si le fournisseur change dynamiquement, l'instance est recréée.
    """
    global _reranker_instance, _reranker_provider

    # Utilisation du fournisseur par défaut si aucun n'est spécifié
    if provider is None:
        provider = RERANK_PROVIDER
    # Normalisation du nom du fournisseur en minuscules sans espaces
    provider = provider.lower().strip()

    # Réutilisation de l'instance existante si le fournisseur n'a pas changé
    if _reranker_instance is not None and _reranker_provider == provider:
        return _reranker_instance

    # Construction du reranker FlashRank (exécution locale, sans API externe)
    if provider == "flashrank":
        # Importation locale du reranker FlashRank
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        # Instanciation avec le modèle et le top-N configurés
        _reranker_instance = FlashrankRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)

    # Construction du reranker Cohere (API cloud)
    elif provider == "cohere":
        # Importation locale du reranker Cohere
        from langchain_cohere import CohereRerank
        _reranker_instance = CohereRerank(
            # Utilisation du modèle configuré ou du modèle par défaut
            model=RERANK_MODEL or "rerank-v3.5",
            # Nombre de documents à conserver après reclassement
            top_n=RERANK_TOP_N,
            # Clé API pour l'authentification Cohere
            cohere_api_key=RERANK_API_KEY,
        )

    # Construction du reranker Jina (API cloud)
    elif provider == "jina":
        # Importation locale du reranker Jina
        from langchain_community.document_compressors.jina_rerank import JinaRerank
        _reranker_instance = JinaRerank(
            # Utilisation du modèle configuré ou du modèle multilingue par défaut
            model=RERANK_MODEL or "jina-reranker-v2-base-multilingual",
            # Nombre de documents à conserver après reclassement
            top_n=RERANK_TOP_N,
            # Clé API pour l'authentification Jina
            jina_api_key=RERANK_API_KEY,
        )

    else:
        # Levée d'une erreur si le fournisseur de reranking n'est pas reconnu
        raise ValueError(
            f"Unknown RERANK_PROVIDER: '{provider}'. "
            "Supported: flashrank, cohere, jina"
        )

    # Mémorisation du fournisseur utilisé pour l'instance en cache
    _reranker_provider = provider
    log.info("Reranker initialized (provider=%s, model=%s)", provider, RERANK_MODEL)
    return _reranker_instance


def _filter_by_score(docs: list[Document], threshold: float) -> list[Document]:
    """Drop documents below the rerank score threshold."""
    # Si le seuil est nul ou négatif, on conserve tous les documents sans filtrage
    if threshold <= 0.0:
        return docs
    return [
        d for d in docs
        # Conservation uniquement des documents dont le score dépasse le seuil
        if d.metadata.get("relevance_score", 1.0) >= threshold
    ]


def rerank(state: RAGState) -> dict:
    """
    Stage 2 — Rerank candidates and keep the top-N most relevant.

    - RERANK_PROVIDER: flashrank, cohere, or jina
    - RERANK_MODEL: model name for the chosen provider
    - RERANK_TOP_N: how many to keep
    - RERANK_SCORE_THRESHOLD: minimum confidence to keep
    """
    # Journalisation de l'état d'entrée pour le débogage
    log.debug("[INPUT] rerank:\n%s", deep_repr(dict(state)))
    # Récupération des documents candidats depuis l'état
    candidates = state.get("candidates", [])
    # Récupération de la question pour le calcul de pertinence
    question = state["question"]

    # Vérification s'il y a des candidats à reclasser
    if not candidates:
        log.warning("No candidates to rerank")
        # Retour d'un contexte vide si aucun candidat disponible
        result = {"context": []}
        log.debug("[OUTPUT] rerank:\n%s", deep_repr(result))
        return result

    # Vérification si le reranking est désactivé dans la configuration
    if not RERANK_ENABLED:
        log.debug("Reranking disabled — passing candidates through as context")
        # Passage direct des candidats comme contexte sans reclassement
        result = {"context": candidates}
        log.debug("[OUTPUT] rerank:\n%s", deep_repr(result))
        return result

    try:
        # Construction de l'instance du reranker selon le fournisseur configuré
        reranker = _build_reranker(RERANK_PROVIDER)
        # Reclassement des candidats par pertinence vis-à-vis de la question
        docs = reranker.compress_documents(candidates, question)
        # Conversion du résultat itérable en liste
        docs = list(docs)
    # Capture de toute exception lors du reranking
    except Exception as e:
        # Journalisation de l'erreur de reranking
        log.error("Reranking failed (provider=%s): %s", RERANK_PROVIDER, e)
        # Repli sur les candidats non reclassés pour que le pipeline puisse quand même répondre
        docs = candidates

    # Filtrage des documents en dessous du seuil de score de pertinence
    docs = _filter_by_score(docs, RERANK_SCORE_THRESHOLD)

    # Extraction des numéros de page des documents retenus
    pages_hit = [d.metadata.get("page", "?") for d in docs]
    # Journalisation du résultat du reranking
    log.info("Reranked → %d docs %s (provider=%s)", len(docs), pages_hit, RERANK_PROVIDER)
    # Construction du résultat avec les documents reclassés comme contexte
    result = {"context": docs}
    # Journalisation du résultat de sortie
    log.debug("[OUTPUT] rerank:\n%s", deep_repr(result))
    return result
