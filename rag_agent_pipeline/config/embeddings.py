"""Embedding models — dense (FastEmbed, local) and sparse (BM25 via FastEmbed)."""

# Module pour accéder aux variables d'environnement
import os

# Modèle d'embedding dense local via FastEmbed
from langchain_community.embeddings import FastEmbedEmbeddings
# Modèle d'embedding sparse (BM25) via FastEmbed
from langchain_qdrant import FastEmbedSparse

# S'assure que le fichier .env est chargé
import config.env  # noqa: F401

# Embeddings denses (exécutés localement via FastEmbed — pas de clé API nécessaire)
# Nom du modèle d'embedding dense
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# Dimension des vecteurs d'embedding
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "384"))

# Instancie le modèle d'embedding dense
EMBEDDINGS = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

# Embeddings sparse pour la recherche hybride (BM25 via FastEmbed, exécuté localement)
# Nom du modèle d'embedding sparse
SPARSE_MODEL      = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
# Instancie le modèle d'embedding sparse BM25
SPARSE_EMBEDDINGS = FastEmbedSparse(model_name=SPARSE_MODEL)
