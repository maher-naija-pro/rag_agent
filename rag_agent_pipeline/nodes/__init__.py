"""Pipeline nodes — one function per file."""

# Importation de chaque noeud du pipeline depuis son module respectif
# Fonction de reconnaissance optique de caractères
from nodes.ocr import ocr_page
# Fonction de chargement et extraction de texte PDF
from nodes.loader import load_pdf
# Fonction de découpage du texte en morceaux
from nodes.chunker import chunk
# Fonction d'extraction de métadonnées
from nodes.metadata import extract_metadata
# Fonction d'encodage vectoriel et stockage
from nodes.embedder import embed_and_store
# Fonctions de cache sémantique
from nodes.cache import cache_check, cache_store
# Réécriture de la requête utilisateur
from nodes.query_rewriter import rewrite_query
# Expansion de la requête avec des termes supplémentaires
from nodes.query_expander import expand_query
# Génération hypothétique de document pour améliorer la recherche
from nodes.hyde import hyde
# Auto-requête avec filtres structurés
from nodes.self_query import self_query
# Recherche de documents pertinents dans la base vectorielle
from nodes.retriever import retrieve
# Réordonnancement des résultats par pertinence
from nodes.reranker import rerank
# Génération de la réponse finale par le LLM
from nodes.generator import generate

# Liste des symboles exportés par ce module
__all__ = [
    "ocr_page", "load_pdf", "chunk", "extract_metadata", "embed_and_store",
    "cache_check", "cache_store",
    "rewrite_query", "expand_query", "hyde", "self_query", "retrieve", "rerank", "generate",
]
