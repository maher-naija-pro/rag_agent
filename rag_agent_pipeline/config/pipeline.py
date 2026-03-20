"""Pipeline tuning parameters — OCR, chunking, retrieval, reranking."""

# Module pour accéder aux variables d'environnement
import os

# S'assure que le fichier .env est chargé
import config.env  # noqa: F401

# OCR (Reconnaissance optique de caractères)
# Langues pour l'OCR (français + anglais par défaut)
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "fra+eng")
# Résolution en DPI pour la conversion PDF vers image
OCR_DPI       = int(os.getenv("OCR_DPI", "300"))

# Découpage en morceaux (chunking)
# Taille maximale de chaque morceau de texte en caractères
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
# Chevauchement entre morceaux consécutifs en caractères
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Recherche (retrieval)
# Nombre de documents à récupérer lors de la recherche
RETRIEVAL_K          = int(os.getenv("RETRIEVAL_K", "15"))
# Seuil minimum de similarité pour garder un résultat
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
# Pondération entre recherche dense (1.0) et sparse (0.0)
HYBRID_FUSION_ALPHA  = float(os.getenv("HYBRID_FUSION_ALPHA", "0.5"))

# Cache sémantique
# Active/désactive le cache sémantique
CACHE_ENABLED              = os.getenv("CACHE_ENABLED", "false").lower() in ("true", "1", "yes")
# Durée de vie du cache en secondes (1 heure par défaut)
CACHE_TTL                  = int(os.getenv("CACHE_TTL", "3600"))
# Seuil de similarité pour considérer un hit de cache
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
# Nombre maximal d'entrées dans le cache
CACHE_MAX_SIZE             = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Réécriture de requête
# Active/désactive la réécriture de requête par le LLM
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "false").lower() in ("true", "1", "yes")

# Expansion de requête
# Active/désactive la génération de requêtes multiples
QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() in ("true", "1", "yes")
# Nombre de variantes de requête à générer
QUERY_EXPANSION_COUNT   = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))

# Auto-requête (extraction de filtres de métadonnées)
# Active/désactive l'extraction automatique de filtres depuis la requête
SELF_QUERY_ENABLED = os.getenv("SELF_QUERY_ENABLED", "false").lower() in ("true", "1", "yes")

# HyDE (Hypothetical Document Embedding)
# Active/désactive la génération d'un document hypothétique pour améliorer la recherche
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "false").lower() in ("true", "1", "yes")

# Extraction de métadonnées
# Active/désactive l'extraction automatique de métadonnées des documents
METADATA_EXTRACTION_ENABLED = os.getenv("METADATA_EXTRACTION_ENABLED", "true").lower() in ("true", "1", "yes")
# Liste des champs de métadonnées à extraire
METADATA_FIELDS = [f.strip() for f in os.getenv("METADATA_FIELDS", "dates,keywords,language").split(",") if f.strip()]

# Reclassement (reranking)
# Active/désactive le reclassement des résultats de recherche
RERANK_ENABLED         = os.getenv("RERANK_ENABLED", "true").lower() in ("true", "1", "yes")
# Fournisseur du modèle de reclassement
RERANK_PROVIDER        = os.getenv("RERANK_PROVIDER", "flashrank")
# Modèle utilisé pour le reclassement
RERANK_MODEL           = os.getenv("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
# Nombre de documents conservés après reclassement
RERANK_TOP_N           = int(os.getenv("RERANK_TOP_N", "4"))
# Score minimum pour qu'un document soit conservé après reclassement
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.1"))
# Clé API pour le service de reclassement (vide si local)
RERANK_API_KEY         = os.getenv("RERANK_API_KEY", "")
