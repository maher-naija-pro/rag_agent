"""Configuration package — re-exports all settings for backward compatibility.

Modules:
    config.env        — .env loader (auto-imported)
    config.llm        — LLM singleton
    config.embeddings — Dense + sparse embedding models
    config.qdrant     — Qdrant client, store, collection
    config.pipeline   — Chunking, retrieval, reranking params
    config.server     — API host/port, CORS, upload, logging

Usage:
    from config import LLM, EMBEDDINGS, get_store
    from config.qdrant import get_client
    from config.pipeline import CHUNK_SIZE
"""

# Réexporte tout pour que `from config import X` continue de fonctionner.

# Importe la configuration des modèles de langage (un par étape du pipeline)
from config.llm import (  # noqa: F401
    # Alias de rétro-compatibilité (= LLM_GENERATE)
    LLM,
    # Instances LLM par étape du pipeline
    LLM_REWRITE,
    LLM_EXPAND,
    LLM_HYDE,
    LLM_SELF_QUERY,
    LLM_GENERATE,
    # Paramètres partagés
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_API_KEY,
)

# Importe la configuration des embeddings
from config.embeddings import (  # noqa: F401
    # Dimension des vecteurs d'embedding denses
    EMBEDDING_DIM,
    # Nom du modèle d'embedding dense
    EMBEDDING_MODEL,
    # Instance du modèle d'embedding dense
    EMBEDDINGS,
    # Instance du modèle d'embedding sparse (BM25)
    SPARSE_EMBEDDINGS,
    # Nom du modèle d'embedding sparse
    SPARSE_MODEL,
)

# Importe la configuration de la base vectorielle Qdrant
from config.qdrant import (  # noqa: F401
    # Nom de la collection Qdrant
    COLLECTION,
    # Clé API pour Qdrant
    QDRANT_API_KEY,
    # URL du serveur Qdrant
    QDRANT_URL,
    # Fonction pour obtenir le client Qdrant (singleton)
    get_client,
    # Fonction pour obtenir le store vectoriel (singleton)
    get_store,
    # Fonction pour remplacer le store vectoriel global
    set_store,
)

# Importe les paramètres du pipeline RAG
from config.pipeline import (  # noqa: F401
    # Chevauchement entre les morceaux de texte
    CHUNK_OVERLAP,
    # Taille des morceaux de texte
    CHUNK_SIZE,
    # Pondération de la fusion hybride dense/sparse
    HYBRID_FUSION_ALPHA,
    # Active/désactive l'extraction de métadonnées
    METADATA_EXTRACTION_ENABLED,
    # Champs de métadonnées à extraire
    METADATA_FIELDS,
    # Active/désactive le cache sémantique
    CACHE_ENABLED,
    # Taille maximale du cache
    CACHE_MAX_SIZE,
    # Seuil de similarité pour le cache
    CACHE_SIMILARITY_THRESHOLD,
    # Durée de vie des entrées du cache en secondes
    CACHE_TTL,
    # Active/désactive HyDE (Hypothetical Document Embedding)
    HYDE_ENABLED,
    # Active/désactive l'auto-requête avec filtres de métadonnées
    SELF_QUERY_ENABLED,
    # Nombre de requêtes générées pour l'expansion
    QUERY_EXPANSION_COUNT,
    # Active/désactive l'expansion de requête
    QUERY_EXPANSION_ENABLED,
    # Active/désactive la réécriture de requête
    QUERY_REWRITE_ENABLED,
    # Résolution DPI pour l'OCR
    OCR_DPI,
    # Langues utilisées pour l'OCR
    OCR_LANGUAGES,
    # Active/désactive le reclassement des résultats
    RERANK_ENABLED,
    # Clé API pour le service de reclassement
    RERANK_API_KEY,
    # Modèle utilisé pour le reclassement
    RERANK_MODEL,
    # Fournisseur du service de reclassement
    RERANK_PROVIDER,
    # Seuil de score minimum pour le reclassement
    RERANK_SCORE_THRESHOLD,
    # Nombre de résultats conservés après reclassement
    RERANK_TOP_N,
    # Nombre de documents à récupérer
    RETRIEVAL_K,
    # Seuil de similarité minimum pour la recherche
    SIMILARITY_THRESHOLD,
)

# Importe la configuration du serveur API
from config.server import (  # noqa: F401
    # Adresse d'écoute du serveur API
    API_HOST,
    # Port d'écoute du serveur API
    API_PORT,
    # Origines autorisées pour les requêtes CORS
    CORS_ORIGINS,
    # Niveau de journalisation
    LOG_LEVEL,
    # Taille maximale d'un fichier en octets
    MAX_FILE_SIZE,
    # Taille maximale d'upload en mégaoctets
    MAX_UPLOAD_SIZE_MB,
    # Répertoire de stockage des fichiers uploadés
    UPLOAD_DIR,
)
