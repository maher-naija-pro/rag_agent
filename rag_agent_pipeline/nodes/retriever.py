"""Node — retrieve: hybrid search (dense + sparse) from Qdrant."""

from __future__ import annotations

# Importation du type Document de LangChain
from langchain_core.documents import Document

from config import (
    # Coefficient alpha pour la fusion hybride dense/sparse
    HYBRID_FUSION_ALPHA,
    # Nombre de documents à récupérer par requête
    RETRIEVAL_K,
    # Seuil de similarité minimum pour filtrer les résultats
    SIMILARITY_THRESHOLD,
    # Fonction pour obtenir le store vectoriel Qdrant
    get_store,
)
# Importation du logger et de la fonction de représentation détaillée
from logger import get_logger, deep_repr
# Importation du type d'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.retriever")


def _build_search_kwargs(source: str, metadata_filter: dict | None = None) -> dict:
    """Build search_kwargs for the Qdrant retriever."""
    # Initialisation avec le nombre de résultats souhaité
    search_kwargs: dict = {"k": RETRIEVAL_K}

    # Application du seuil de similarité si défini (supérieur à zéro)
    if SIMILARITY_THRESHOLD > 0.0:
        search_kwargs["score_threshold"] = SIMILARITY_THRESHOLD

    # Modification de l'alpha de fusion si différent de la valeur par défaut (0.5)
    if HYBRID_FUSION_ALPHA != 0.5:
        search_kwargs["alpha"] = HYBRID_FUSION_ALPHA

    # Construction des conditions de filtre Qdrant
    # Initialisation de la liste des conditions de filtrage
    conditions = []

    # Filtrage par source (nom du fichier PDF) si un nom de source est spécifié
    if source:
        # Importation locale des modèles Qdrant
        from qdrant_client.models import FieldCondition, MatchValue
        conditions.append(
            # Condition de correspondance exacte sur le champ source
            FieldCondition(key="metadata.source", match=MatchValue(value=source))
        )
        log.debug("Filtering retrieval to source='%s'", source)

    # Application des filtres de métadonnées extraits par le noeud self-query
    # Vérification si des filtres de métadonnées existent
    if metadata_filter:
        # Importation locale des modèles Qdrant
        from qdrant_client.models import FieldCondition, MatchValue
        # Parcours de chaque filtre extrait
        for key, value in metadata_filter.items():
            # Filtre par numéro de page (entier)
            if key == "page" and isinstance(value, int):
                conditions.append(
                    # Condition de correspondance exacte sur la page
                    FieldCondition(key="metadata.page", match=MatchValue(value=value))
                )
            # Filtre par langue du document (chaîne)
            elif key == "language" and isinstance(value, str):
                conditions.append(
                    # Condition de correspondance exacte sur la langue
                    FieldCondition(key="metadata.language", match=MatchValue(value=value))
                )
            # Filtre par présence de tableaux (booléen)
            elif key == "has_tables" and isinstance(value, bool):
                conditions.append(
                    # Condition de correspondance exacte sur les tableaux
                    FieldCondition(key="metadata.has_tables", match=MatchValue(value=value))
                )
            else:
                # Journalisation des filtres non supportés ignorés
                log.debug("Skipping unsupported filter: %s=%s", key, value)
        # Journalisation du nombre de filtres appliqués
        log.info("Applied %d metadata filters: %s", len(metadata_filter), metadata_filter)

    # Si des conditions de filtrage ont été construites
    if conditions:
        # Importation locale du modèle Filter de Qdrant
        from qdrant_client.models import Filter
        # Création du filtre combinant toutes les conditions avec un ET logique
        search_kwargs["filter"] = Filter(must=conditions)

    # Retour des paramètres de recherche construits
    return search_kwargs


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate documents by page_content."""
    # Ensemble pour suivre les contenus déjà rencontrés
    seen: set[str] = set()
    # Liste pour stocker les documents uniques
    unique: list[Document] = []
    # Parcours de tous les documents candidats
    for doc in docs:
        # Utilisation du contenu textuel comme clé de déduplication
        key = doc.page_content
        # Vérification si le contenu n'a pas déjà été rencontré
        if key not in seen:
            # Marquage du contenu comme vu
            seen.add(key)
            # Ajout du document à la liste des uniques
            unique.append(doc)
    # Retour de la liste des documents dédupliqués
    return unique


def retrieve(state: RAGState) -> dict:
    """
    Hybrid search (dense + BM25 sparse) → candidates from Qdrant.

    When expanded_queries is populated (query expansion enabled),
    runs retrieval for each variant and merges + deduplicates results.
    Otherwise uses the single question.
    """
    # Journalisation de l'état d'entrée pour le débogage
    log.debug("[INPUT] retrieve:\n%s", deep_repr(dict(state)))
    # Récupération des requêtes étendues si disponibles
    queries = state.get("expanded_queries", [])
    # Si aucune requête étendue n'est disponible
    if not queries:
        # Utilisation de la question unique comme requête de recherche
        queries = [state["question"]]

    # Récupération du filtre de source (nom du fichier PDF ciblé)
    source = state.get("source", "")
    # Récupération des filtres de métadonnées extraits par self-query
    metadata_filter = state.get("metadata_filter", {})
    # Construction des paramètres de recherche Qdrant
    search_kwargs = _build_search_kwargs(source, metadata_filter)

    try:
        # Création du retriever à partir du store vectoriel Qdrant
        base_retriever = get_store().as_retriever(
            # Type de recherche par similarité vectorielle
            search_type="similarity",
            # Application des paramètres de recherche construits
            search_kwargs=search_kwargs,
        )

        # Initialisation de la liste de tous les documents candidats
        all_candidates: list[Document] = []
        # Parcours de chaque requête (originale + variantes + hypothétique)
        for query in queries:
            # Exécution de la recherche vectorielle pour chaque requête
            results = base_retriever.invoke(query)
            # Ajout des résultats à la liste globale des candidats
            all_candidates.extend(results)

        # Déduplication des résultats fusionnés de toutes les requêtes
        candidates = _deduplicate(all_candidates)

    # Capture de toute exception lors de la recherche
    except Exception as e:
        # Journalisation de l'erreur de recherche
        log.error("Retrieval failed: %s", e)
        # Retour d'une liste vide en cas d'erreur
        result = {"candidates": []}
        log.debug("[OUTPUT] retrieve:\n%s", deep_repr(result))
        return result

    # Extraction des numéros de page des candidats trouvés
    pages_hit = [d.metadata.get("page", "?") for d in candidates]
    # Journalisation des statistiques de recherche
    log.info(
        "Retrieved %d candidates (%d queries, %d before dedup) %s",
        len(candidates), len(queries), len(all_candidates), pages_hit,
    )
    # Construction du résultat avec les documents candidats
    result = {"candidates": candidates}
    # Journalisation du résultat de sortie
    log.debug("[OUTPUT] retrieve:\n%s", deep_repr(result))
    return result
