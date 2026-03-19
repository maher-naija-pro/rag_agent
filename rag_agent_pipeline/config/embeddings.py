"""Embedding models — dense (FastEmbed, local) and sparse (BM25 via FastEmbed)."""

import os

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse

import config.env  # noqa: F401  ensure .env is loaded

# Dense embeddings (local via FastEmbed — no API key needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "384"))

EMBEDDINGS = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

# Sparse embeddings for hybrid search (BM25 via FastEmbed, runs locally)
SPARSE_MODEL      = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
SPARSE_EMBEDDINGS = FastEmbedSparse(model_name=SPARSE_MODEL)
