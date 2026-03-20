"""RAG pipeline state definition."""

# Permet d'utiliser les annotations de type différées (PEP 604)
from __future__ import annotations

# Importation des types pour les annotations et les dictionnaires typés
from typing import Annotated, TypedDict

# Classe représentant un document avec contenu et métadonnées
from langchain_core.documents import Document
# Classe de base pour les messages du modèle de langage
from langchain_core.messages import BaseMessage
# Fonction réductrice qui accumule les messages dans la liste
from langgraph.graph.message import add_messages


# Définition de l'état partagé du pipeline RAG sous forme de dictionnaire typé
class RAGState(TypedDict):
    # Historique des messages de la conversation, accumulés via add_messages
    messages:   Annotated[list[BaseMessage], add_messages]
    # Dernière question de l'utilisateur (potentiellement reformulée)
    question:   str
    # Question originale de l'utilisateur avant toute réécriture
    original_question: str
    # Requêtes variantes générées par l'expansion de la question
    expanded_queries: list[str]
    # Filtres de métadonnées extraits par auto-interrogation (ex. : {"page": 5, "language": "fr"})
    metadata_filter: dict
    # Nom du fichier PDF — utilisé pour filtrer la recherche sur ce document
    source:     str
    # Pages brutes extraites du chargeur PDF
    raw_pages:  list[Document]
    # Fragments de texte issus du découpage
    chunks:     list[Document]
    # Résultats bruts de la recherche (avant reclassement)
    candidates: list[Document]
    # Résultats de la recherche après reclassement
    context:    list[Document]
    # Réponse finale générée par le modèle de langage
    answer:     str
    # Vrai si la réponse a été servie depuis le cache sémantique
    cache_hit:  bool
    # Vrai une fois que le PDF a été vectorisé et indexé
    ingested:   bool
