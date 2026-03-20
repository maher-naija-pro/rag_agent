"""Node — semantic cache: skip pipeline for repeated/similar questions."""

from __future__ import annotations

# Module pour la gestion du temps
import time
# Décorateurs pour créer des classes de données
from dataclasses import dataclass, field

# Classe de message IA de LangChain
from langchain_core.messages import AIMessage

from config.pipeline import (
    # Activation/désactivation du cache
    CACHE_ENABLED,
    # Nombre maximal d'entrées dans le cache
    CACHE_MAX_SIZE,
    # Seuil de similarité cosinus pour un hit de cache
    CACHE_SIMILARITY_THRESHOLD,
    # Durée de vie des entrées du cache en secondes
    CACHE_TTL,
)
# Modèle d'embeddings pour encoder les questions
from config.embeddings import EMBEDDINGS
# Utilitaires de journalisation
from logger import get_logger, deep_repr
# Type représentant l'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.cache")


@dataclass
class CacheEntry:
    """Structure de données représentant une entrée du cache."""
    # Vecteur d'embedding de la question
    question_embedding: list[float]
    # Réponse générée associée
    answer: str
    # Pages sources utilisées pour la réponse
    source_pages: list[int]
    # Horodatage de création de l'entrée
    timestamp: float


class SemanticCache:
    """In-memory semantic cache with cosine similarity matching."""

    def __init__(self) -> None:
        # Liste interne des entrées du cache
        self._entries: list[CacheEntry] = []

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calcul de la similarité cosinus entre deux vecteurs."""
        # Produit scalaire des deux vecteurs
        dot = sum(x * y for x, y in zip(a, b))
        # Norme euclidienne du vecteur a
        norm_a = sum(x * x for x in a) ** 0.5
        # Norme euclidienne du vecteur b
        norm_b = sum(x * x for x in b) ** 0.5
        # Protection contre la division par zéro
        if norm_a == 0 or norm_b == 0:
            return 0.0
        # Retour de la similarité cosinus
        return dot / (norm_a * norm_b)

    def lookup(self, question_embedding: list[float]) -> CacheEntry | None:
        """Find a cached entry with similarity above threshold."""
        # Récupération du temps actuel
        now = time.time()
        # Meilleure entrée trouvée
        best_entry = None
        # Meilleur score de similarité
        best_score = 0.0

        # Parcours de toutes les entrées du cache
        for entry in self._entries:
            # Ignorer les entrées expirées
            if now - entry.timestamp > CACHE_TTL:
                continue
            # Calcul de la similarité
            score = self._cosine_similarity(question_embedding, entry.question_embedding)
            # Mise à jour si meilleur score trouvé
            if score > best_score:
                best_score = score
                best_entry = entry

        # Vérification du seuil de similarité
        if best_entry and best_score >= CACHE_SIMILARITY_THRESHOLD:
            # Journalisation du hit de cache
            log.info("Cache hit (similarity=%.3f)", best_score)
            return best_entry

        # Aucune entrée suffisamment similaire trouvée
        return None

    def store(self, question_embedding: list[float], answer: str, source_pages: list[int]) -> None:
        """Store a new cache entry, evicting oldest if at capacity."""
        # Récupération du temps actuel
        now = time.time()

        # Suppression des entrées expirées
        self._entries = [e for e in self._entries if now - e.timestamp <= CACHE_TTL]

        # Suppression des entrées les plus anciennes si capacité maximale atteinte
        if len(self._entries) >= CACHE_MAX_SIZE:
            # Tri par horodatage croissant
            self._entries.sort(key=lambda e: e.timestamp)
            # Conservation des entrées les plus récentes
            self._entries = self._entries[-(CACHE_MAX_SIZE - 1):]

        # Ajout de la nouvelle entrée au cache
        self._entries.append(CacheEntry(
            question_embedding=question_embedding,
            answer=answer,
            source_pages=source_pages,
            timestamp=now,
        ))
        # Journalisation du nombre d'entrées
        log.debug("Cached answer (%d entries total)", len(self._entries))

    def clear(self) -> None:
        """Suppression de toutes les entrées du cache."""
        # Sauvegarde du nombre d'entrées avant suppression
        count = len(self._entries)
        # Vidage de la liste des entrées
        self._entries.clear()
        # Journalisation du nombre d'entrées supprimées
        log.info("Cache cleared (%d entries removed)", count)

    @property
    def size(self) -> int:
        """Retourne le nombre d'entrées actuellement dans le cache."""
        return len(self._entries)


# Instance singleton globale — persiste entre les requêtes dans le même processus
_cache = SemanticCache()


def get_cache() -> SemanticCache:
    """Return the global cache instance."""
    # Retour de l'instance unique du cache
    return _cache


def cache_check(state: RAGState) -> dict:
    """
    Check if a similar question was already answered.

    - Embeds the question and searches the cache by cosine similarity
    - If hit: returns cached answer, sets cache_hit=True (skips rest of pipeline)
    - If miss: sets cache_hit=False (pipeline continues normally)
    - Skipped when CACHE_ENABLED=false
    """
    # Journalisation de l'état d'entrée
    log.debug("[INPUT] cache_check:\n%s", deep_repr(dict(state)))
    # Vérification si le cache est activé
    if not CACHE_ENABLED:
        result = {"cache_hit": False}
        log.debug("[OUTPUT] cache_check:\n%s", deep_repr(result))
        # Retour immédiat si cache désactivé
        return result

    # Récupération de la question depuis l'état
    question = state["question"]

    try:
        # Encodage vectoriel de la question
        embedding = EMBEDDINGS.embed_query(question)
        # Recherche dans le cache par similarité
        entry = _cache.lookup(embedding)

        # Si une entrée similaire a été trouvée
        if entry:
            result = {
                # Indication d'un hit de cache
                "cache_hit": True,
                # Réponse mise en cache
                "answer": entry.answer,
                # Contexte vide car réponse depuis le cache
                "context": [],
            }
            log.debug("[OUTPUT] cache_check:\n%s", deep_repr(result))
            return result

    except Exception as e:
        # Avertissement en cas d'erreur de vérification
        log.warning("Cache check failed: %s", e)

    # Aucun hit — le pipeline continue normalement
    result = {"cache_hit": False}
    log.debug("[OUTPUT] cache_check:\n%s", deep_repr(result))
    return result


def cache_store(state: RAGState) -> dict:
    """
    Store the answer in the semantic cache after generation.

    - Only stores if CACHE_ENABLED=true and answer was not from cache
    - Embeds the original question for future lookups
    """
    # Journalisation de l'état d'entrée
    log.debug("[INPUT] cache_store:\n%s", deep_repr(dict(state)))
    # Vérification si le cache est activé
    if not CACHE_ENABLED:
        log.debug("[OUTPUT] cache_store:\n%s", deep_repr({}))
        return {}

    # Ne pas stocker si la réponse vient déjà du cache
    if state.get("cache_hit", False):
        log.debug("[OUTPUT] cache_store:\n%s", deep_repr({}))
        return {}

    # Récupération de la question originale
    question = state.get("original_question") or state["question"]
    # Récupération de la réponse générée
    answer = state.get("answer", "")

    # Ne pas stocker si aucune réponse n'a été générée
    if not answer:
        log.debug("[OUTPUT] cache_store:\n%s", deep_repr({}))
        return {}

    try:
        # Encodage vectoriel de la question pour le cache
        embedding = EMBEDDINGS.embed_query(question)
        # Extraction des numéros de pages sources uniques et triés
        source_pages = sorted(set(
            d.metadata.get("page", 0)
            for d in state.get("context", [])
            if d.metadata.get("page")
        ))
        # Stockage de l'entrée dans le cache
        _cache.store(embedding, answer, source_pages)
    except Exception as e:
        # Avertissement en cas d'erreur de stockage
        log.warning("Cache store failed: %s", e)

    # Journalisation de l'état de sortie
    log.debug("[OUTPUT] cache_store:\n%s", deep_repr({}))
    # Retour d'un dictionnaire vide (pas de modification d'état)
    return {}
