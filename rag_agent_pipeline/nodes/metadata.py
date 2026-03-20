"""Node 3 — extract_metadata: enrich chunks with structured metadata."""

from __future__ import annotations

# Module pour les expressions régulières
import re
# Compteur pour le calcul de fréquence des mots
from collections import Counter

# Classe représentant un document avec contenu et métadonnées
from langchain_core.documents import Document

# Paramètres d'activation et champs de métadonnées
from config.pipeline import METADATA_EXTRACTION_ENABLED, METADATA_FIELDS
# Utilitaires de journalisation
from logger import get_logger, deep_repr
# Type représentant l'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.metadata")

# ── Expressions régulières pour l'extraction de métadonnées ──────────────────

_DATE_PATTERNS = [
    # Format ISO : 2024-01-15
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    # Format européen : 15/01/2024 ou 15-01-2024
    re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b"),
    # Format texte anglais : January 15, 2024 / 15 January 2024 / Jan 2024
    re.compile(
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Oct|Nov|Dec)\s+\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    # Format texte français : 15 janvier 2024
    re.compile(
        r"\b(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|"
        r"août|septembre|octobre|novembre|décembre)\s+\d{4})\b",
        re.IGNORECASE,
    ),
]

# Regex pour détecter les adresses email
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")

# Regex pour détecter les numéros de téléphone (formats variés)
_PHONE_PATTERN = re.compile(
    r"\b(?:\+\d{1,3}[\s\-]?)?"
    r"(?:\(?\d{1,4}\)?[\s\-]?)?"
    r"\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{2,4}\b"
)

# Regex pour détecter les URLs HTTP/HTTPS
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")


def _extract_dates(text: str) -> list[str]:
    """Extract date strings from text."""
    # Liste pour stocker les dates trouvées
    dates = []
    # Parcours de chaque patron de date
    for pattern in _DATE_PATTERNS:
        # Ajout de toutes les correspondances trouvées
        dates.extend(pattern.findall(text))
    # Retour des dates uniques triées
    return sorted(set(dates))


def _extract_emails(text: str) -> list[str]:
    """Extract email addresses from text."""
    # Retour des emails uniques triés
    return sorted(set(_EMAIL_PATTERN.findall(text)))


def _extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    # Retour des URLs uniques triées
    return sorted(set(_URL_PATTERN.findall(text)))


def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract top-N most frequent meaningful words (>4 chars) as keywords."""
    # Extraction des mots de 5 lettres ou plus en minuscules
    words = re.findall(r"\b[a-zA-ZÀ-ÿ]{5,}\b", text.lower())
    # Filtrage des mots vides courants en anglais et français
    stop = {
        "about", "after", "again", "being", "below", "between", "could",
        "didn", "doing", "during", "every", "first", "found", "given",
        "great", "having", "their", "there", "these", "thing", "think",
        "those", "three", "through", "under", "until", "using", "value",
        "where", "which", "while", "would", "would", "years",
        "cette", "comme", "dans", "elles", "entre", "leurs", "mais",
        "notre", "nous", "notre", "autres", "aussi", "avant", "avoir",
        "cette", "comme", "depuis", "encore", "entre", "faire", "leurs",
        "même", "notre", "notre", "peut", "plus", "pour", "quand",
        "quelque", "sont", "tout", "très", "votre",
    }
    # Suppression des mots vides de la liste
    filtered = [w for w in words if w not in stop]
    # Comptage des occurrences de chaque mot
    counts = Counter(filtered)
    # Retour des N mots les plus fréquents
    return [word for word, _ in counts.most_common(top_n)]


def _detect_language(text: str) -> str:
    """Simple language detection based on common word frequency."""
    # Prise d'un échantillon de 2000 caractères en minuscules
    sample = text[:2000].lower()

    # Ensemble de marqueurs typiques du français
    fr_markers = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "en", "que", "pour", "dans", "sur"}
    # Ensemble de marqueurs typiques de l'anglais
    en_markers = {"the", "is", "are", "was", "were", "of", "and", "to", "in", "for", "that", "with", "on", "at"}

    # Extraction de tous les mots uniques de l'échantillon
    words = set(re.findall(r"\b\w+\b", sample))
    # Comptage des marqueurs français trouvés
    fr_score = len(words & fr_markers)
    # Comptage des marqueurs anglais trouvés
    en_score = len(words & en_markers)

    # Si le score français est supérieur
    if fr_score > en_score:
        return "fr"
    # Si le score anglais est supérieur
    if en_score > fr_score:
        return "en"
    # Langue indéterminée si scores égaux
    return "unknown"


def _has_tables(text: str) -> bool:
    """Detect if text likely contains table-like structures."""
    # Découpage du texte en lignes
    lines = text.split("\n")
    # Comptage des lignes avec des séparateurs tabulaires
    tabular_lines = sum(1 for line in lines if line.count("|") >= 2 or line.count("\t") >= 2)
    # Retour vrai si au moins 3 lignes tabulaires détectées
    return tabular_lines >= 3


def _extract_for_chunk(doc: Document, fields: set[str]) -> dict:
    """Extract requested metadata fields for a single chunk."""
    # Récupération du contenu textuel du document
    text = doc.page_content
    # Dictionnaire pour stocker les métadonnées extraites
    meta = {}

    # Extraction des dates si demandé
    if "dates" in fields:
        dates = _extract_dates(text)
        if dates:
            meta["dates"] = dates

    # Extraction des emails si demandé
    if "emails" in fields:
        emails = _extract_emails(text)
        if emails:
            meta["emails"] = emails

    # Extraction des URLs si demandé
    if "urls" in fields:
        urls = _extract_urls(text)
        if urls:
            meta["urls"] = urls

    # Extraction des mots-clés si demandé
    if "keywords" in fields:
        keywords = _extract_keywords(text)
        if keywords:
            meta["keywords"] = keywords

    # Détection de la langue si demandé
    if "language" in fields:
        meta["language"] = _detect_language(text)

    # Détection de tableaux si demandé
    if "has_tables" in fields:
        meta["has_tables"] = _has_tables(text)

    # Comptage de caractères si demandé
    if "char_count" in fields:
        meta["char_count"] = len(text)

    # Retour des métadonnées extraites
    return meta


def extract_metadata(state: RAGState) -> dict:
    """
    Enrich each chunk with structured metadata extracted from its content.

    Controlled by METADATA_FIELDS env var (comma-separated).
    Available fields: dates, emails, urls, keywords, language, has_tables, char_count.
    """
    # Journalisation de l'état d'entrée
    log.debug("[INPUT] extract_metadata:\n%s", deep_repr(dict(state)))
    # Récupération des morceaux depuis l'état
    chunks = state.get("chunks", [])

    # Vérification si l'extraction de métadonnées est activée
    if not METADATA_EXTRACTION_ENABLED:
        log.debug("Metadata extraction disabled — passing through")
        # Retour des morceaux sans modification
        result = {"chunks": chunks}
        log.debug("[OUTPUT] extract_metadata:\n%s", deep_repr(result))
        return result

    # Vérification qu'il y a des morceaux à traiter
    if not chunks:
        log.warning("No chunks to extract metadata from")
        result = {"chunks": []}
        log.debug("[OUTPUT] extract_metadata:\n%s", deep_repr(result))
        return result

    # Conversion de la liste des champs en ensemble pour recherche rapide
    fields = set(METADATA_FIELDS)
    # Vérification que des champs sont configurés
    if not fields:
        log.info("METADATA_FIELDS is empty — skipping metadata extraction")
        result = {"chunks": chunks}
        log.debug("[OUTPUT] extract_metadata:\n%s", deep_repr(result))
        return result

    # Liste pour stocker les morceaux enrichis
    enriched = []
    # Statistiques d'extraction par champ
    stats: dict[str, int] = {f: 0 for f in fields}

    # Parcours de chaque morceau
    for doc in chunks:
        # Extraction des métadonnées pour ce morceau
        extracted = _extract_for_chunk(doc, fields)

        # Fusion des métadonnées existantes avec les nouvelles
        new_meta = {**doc.metadata, **extracted}
        # Création d'un nouveau document enrichi
        enriched.append(Document(
            page_content=doc.page_content,
            metadata=new_meta,
        ))

        # Mise à jour des statistiques pour chaque champ extrait
        for key in extracted:
            stats[key] = stats.get(key, 0) + 1

    # Journalisation du résumé de l'extraction
    log.info(
        "Metadata extracted for %d chunks: %s",
        len(enriched),
        ", ".join(f"{k}={v}" for k, v in stats.items() if v > 0),
    )
    # Construction du résultat
    result = {"chunks": enriched}
    # Journalisation de l'état de sortie
    log.debug("[OUTPUT] extract_metadata:\n%s", deep_repr(result))
    # Retour des morceaux enrichis
    return result
