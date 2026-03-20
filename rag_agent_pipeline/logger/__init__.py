"""Centralized logging configuration — stdout only.

Usage in any module:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""

# Permet les annotations de type différées
from __future__ import annotations

# Module pour la sérialisation/désérialisation JSON
import json
# Module standard de journalisation Python
import logging
# Accès aux flux système (stdout, stderr)
import sys
# Type générique pour les annotations
from typing import Any

# Importe le niveau de log depuis la configuration serveur
from config.server import LOG_LEVEL


def _configure_root() -> None:
    """Set up the root logger once (idempotent)."""
    # Récupère le logger racine de l'espace de noms "rag"
    root = logging.getLogger("rag")
    if root.handlers:
        # Déjà configuré, on ne reconfigure pas (idempotent)
        return

    # Définit le niveau de log (INFO par défaut)
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Crée un handler qui écrit sur la sortie standard
    handler = logging.StreamHandler(sys.stdout)
    # Aligne le niveau du handler sur celui du logger
    handler.setLevel(root.level)

    # Crée le format d'affichage des messages de log
    formatter = logging.Formatter(
        # Format : horodatage | niveau | nom | message
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        # Format de la date
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Applique le formateur au handler
    handler.setFormatter(formatter)
    # Attache le handler au logger racine
    root.addHandler(handler)


# Exécute la configuration du logger au chargement du module
_configure_root()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'rag' namespace."""
    # Retourne un logger enfant sous l'espace "rag"
    return logging.getLogger(f"rag.{name}")


# ── Deep state serialisation for input/output logging ────────────────────────

# Nombre maximum de caractères affichés pour le contenu
_CONTENT_TRUNCATE = 200


def _summarise_value(value: Any) -> Any:
    """Return a JSON-friendly summary of a single value."""
    # Document LangChain
    # Vérifie si c'est un Document LangChain
    if hasattr(value, "page_content") and hasattr(value, "metadata"):
        # Récupère le contenu textuel du document
        content = value.page_content
        # Tronque le contenu si trop long
        preview = content[:_CONTENT_TRUNCATE] + ("…" if len(content) > _CONTENT_TRUNCATE else "")
        # Retourne un résumé du document
        return {"_doc": preview, "meta": value.metadata, "chars": len(content)}

    # Messages LangChain / LangGraph
    # Vérifie si c'est un message LangChain
    if hasattr(value, "content") and hasattr(value, "type"):
        # Convertit le contenu du message en chaîne
        content = str(value.content)
        # Tronque si nécessaire
        preview = content[:_CONTENT_TRUNCATE] + ("…" if len(content) > _CONTENT_TRUNCATE else "")
        # Retourne un résumé du message
        return {"_msg": value.type, "content": preview, "chars": len(content)}

    # Liste d'éléments — résume chaque élément
    # Vérifie si la valeur est une liste
    if isinstance(value, list):
        if len(value) == 0:
            # Retourne une liste vide si pas d'éléments
            return []
        # Pour les grandes listes de nombres (vecteurs d'embeddings), affiche seulement la taille
        # Vérifie si c'est un vecteur numérique
        if all(isinstance(v, (int, float)) for v in value):
            # Retourne la taille du vecteur au lieu de tout afficher
            return f"<vector len={len(value)}>"
        # Résume chaque élément récursivement
        return [_summarise_value(v) for v in value]

    # Dictionnaire — parcours récursif
    # Vérifie si la valeur est un dictionnaire
    if isinstance(value, dict):
        # Résume chaque valeur du dictionnaire
        return {k: _summarise_value(v) for k, v in value.items()}

    # Retourne la valeur telle quelle pour les types simples
    return value


def deep_repr(data: dict) -> str:
    """Return a pretty JSON string summarising a state or output dict."""
    # Résume chaque valeur du dictionnaire d'entrée
    summarised = {k: _summarise_value(v) for k, v in data.items()}
    try:
        # Sérialise en JSON formaté
        return json.dumps(summarised, ensure_ascii=False, indent=2, default=str)
    except Exception:
        # En cas d'erreur, retourne la représentation en chaîne brute
        return str(summarised)
