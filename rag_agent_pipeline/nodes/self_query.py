"""Node — self_query: extract metadata filters from the user's question."""

from __future__ import annotations

# Importation du module JSON pour le parsing des filtres
import json
# Importation du module d'expressions régulières
import re

# Importation des types de messages LangChain
from langchain_core.messages import HumanMessage, SystemMessage

# Importation du modèle de langage dédié à l'extraction de filtres
from config import LLM_SELF_QUERY as LLM
# Paramètres de configuration du self-query
from config.pipeline import SELF_QUERY_ENABLED, METADATA_FIELDS
# Importation du logger et de la fonction de représentation détaillée
from logger import get_logger, deep_repr
# Importation du type d'état du pipeline RAG
from state import RAGState

# Initialisation du logger pour ce module
log = get_logger("nodes.self_query")

# Prompt système pour extraire les filtres de métadonnées depuis la question
SELF_QUERY_PROMPT = """\
You are a metadata filter extractor for a document search system.
Given a user question, extract any structured filters that can narrow down the search.

Available metadata fields on each document chunk:
{fields_desc}

Rules:
- Output ONLY a JSON object with the extracted filters. No explanation.
- Only include fields you are confident about. Omit uncertain fields.
- If no filters can be extracted, output: {{}}
- For "page", extract the exact page number as an integer.
- For "language", use ISO codes: "fr" for French, "en" for English.
- For "has_tables", use true/false.
- For "dates", extract date strings as they might appear (e.g. "2024-01", "2024-01-15").

Examples:
- "What's on page 5?" → {{"page": 5}}
- "Summarize the French parts" → {{"language": "fr"}}
- "Show me the tables" → {{"has_tables": true}}
- "What happened in January 2024?" → {{"dates": "2024-01"}}
- "What is the main topic?" → {{}}
"""

# Dictionnaire associant chaque champ de métadonnées à sa description pour le prompt
_FIELD_DESCRIPTIONS = {
    "dates": "dates (list[str]) — date strings found in the chunk",
    "emails": "emails (list[str]) — email addresses found in the chunk",
    "urls": "urls (list[str]) — URLs found in the chunk",
    "keywords": "keywords (list[str]) — top frequent meaningful words",
    "language": 'language (str) — detected language: "fr", "en", or "unknown"',
    "has_tables": "has_tables (bool) — whether the chunk contains table-like structures",
    "char_count": "char_count (int) — character count of the chunk",
    "page": "page (int) — page number in the source PDF",
    "source": "source (str) — PDF filename",
}


def _build_fields_description(fields: list[str]) -> str:
    """Build the fields description for the prompt based on configured METADATA_FIELDS."""
    # Ajout systématique de page et source aux champs disponibles
    all_fields = set(fields) | {"page", "source"}
    # Initialisation de la liste des lignes de description
    lines = []
    # Parcours des champs triés par ordre alphabétique
    for f in sorted(all_fields):
        # Vérification que le champ a une description connue
        if f in _FIELD_DESCRIPTIONS:
            # Ajout de la description formatée à la liste
            lines.append(f"- {_FIELD_DESCRIPTIONS[f]}")
    # Concaténation de toutes les descriptions en une seule chaîne
    return "\n".join(lines)


def _parse_filter_response(response: str) -> dict:
    """Parse the LLM JSON response into a filter dict, handling edge cases."""
    # Nettoyage des espaces autour de la réponse
    text = response.strip()

    # Suppression des blocs de code markdown si présents
    # Détection d'un bloc de code markdown
    if text.startswith("```"):
        # Retrait de l'ouverture du bloc de code
        text = re.sub(r"^```(?:json)?\s*", "", text)
        # Retrait de la fermeture du bloc de code
        text = re.sub(r"\s*```$", "", text)

    try:
        # Tentative de parsing JSON de la réponse
        result = json.loads(text)
        # Vérification que le résultat est bien un dictionnaire
        if isinstance(result, dict):
            # Retour du dictionnaire de filtres extraits
            return result
    # Capture des erreurs de parsing JSON
    except json.JSONDecodeError:
        # Ignoré silencieusement, on retourne un dictionnaire vide
        pass

    # Retour d'un dictionnaire vide si le parsing échoue
    return {}


def self_query(state: RAGState) -> dict:
    """
    Extract structured metadata filters from the user's question.

    - Analyzes the question for references to pages, dates, language, etc.
    - Stores extracted filters in metadata_filter
    - The retriever node uses these filters to pre-filter Qdrant results
    - Skipped when SELF_QUERY_ENABLED=false
    """
    # Journalisation de l'état d'entrée pour le débogage
    log.debug("[INPUT] self_query:\n%s", deep_repr(dict(state)))
    # Vérification si le self-query est désactivé dans la configuration
    if not SELF_QUERY_ENABLED:
        log.debug("Self-query disabled — no metadata filters")
        # Retour d'un filtre vide
        result = {"metadata_filter": {}}
        log.debug("[OUTPUT] self_query:\n%s", deep_repr(result))
        return result

    # Récupération de la question de l'utilisateur
    question = state["question"]
    # Construction de la description des champs de métadonnées disponibles
    fields_desc = _build_fields_description(METADATA_FIELDS)

    # Construction du prompt pour le LLM
    prompt = [
        # Instruction système avec la description des champs injectée
        SystemMessage(content=SELF_QUERY_PROMPT.format(fields_desc=fields_desc)),
        # Question de l'utilisateur à analyser
        HumanMessage(content=question),
    ]

    try:
        # Appel au LLM et nettoyage de la réponse
        response = LLM.invoke(prompt).content.strip()
        # Parsing de la réponse JSON en dictionnaire de filtres
        filters = _parse_filter_response(response)

        # Vérification si des filtres ont été extraits avec succès
        if filters:
            # Journalisation des filtres extraits
            log.info("Self-query extracted filters: %s", filters)
        else:
            # Journalisation de l'absence de filtres
            log.debug("Self-query: no filters extracted")

        # Construction du résultat avec les filtres de métadonnées
        result = {"metadata_filter": filters}
        # Journalisation du résultat de sortie
        log.debug("[OUTPUT] self_query:\n%s", deep_repr(result))
        return result

    # Capture de toute exception lors de l'extraction des filtres
    except Exception as e:
        # Journalisation de l'erreur
        log.warning("Self-query failed: %s — no filters", e)
        # Retour d'un filtre vide en cas d'erreur
        result = {"metadata_filter": {}}
        log.debug("[OUTPUT] self_query:\n%s", deep_repr(result))
        return result
