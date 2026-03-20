"""Graph definition — wire all nodes together."""

# Permet d'utiliser les annotations de type comme des chaînes
from __future__ import annotations

# Import du type Literal pour restreindre les valeurs de retour
from typing import Literal

# Import des constantes de début/fin et du constructeur de graphe
from langgraph.graph import END, START, StateGraph
# Import du système de sauvegarde de points de contrôle en mémoire
from langgraph.checkpoint.memory import InMemorySaver

# Import de la fonction utilitaire de journalisation
from logger import get_logger
# Import de la définition de l'état partagé du pipeline RAG
from state import RAGState
from nodes import (
    load_pdf, chunk, extract_metadata, embed_and_store,
    rewrite_query, expand_query, hyde, self_query,
    retrieve, rerank, generate,
)

# Initialisation du logger pour ce module
log = get_logger("graph")


def should_ingest(state: RAGState) -> Literal["load_pdf", "rewrite_query"]:
    """
    On the first turn the PDF has not been ingested yet → route to load_pdf.
    On subsequent turns the store is already populated  → skip to rewrite_query.
    """
    if not state.get("ingested", False):
        log.info("First turn → routing to load_pdf")
        return "load_pdf"
    log.debug("Subsequent turn → routing to rewrite_query")
    return "rewrite_query"


# Fonction principale de construction du graphe d'exécution
def build_graph(checkpointer: InMemorySaver):
    """
    Graph topology:

        START
          │
          ▼
      [router] ──── first turn ────► load_pdf → chunk → extract_metadata → embed_and_store → END
          │
          └──── subsequent turns ──► cache_check
                                                                                             │
                                                                              ┌── HIT ──────┤
                                                                              │              └── MISS ──┐
                                                                              │                          ▼
                                                                              │                    rewrite_query
                                                                              │                          │
                                                                              │                     expand_query
                                                                              │                          │
                                                                              │                        hyde
                                                                              │                          │
                                                                              │                     self_query
                                                                              │                          │
                                                                              │                      retrieve
                                                                              │                          │
                                                                              │                       rerank
                                                                              │                          │
                                                                              └────────────► generate
                                                                                                 │
                                                                                            cache_store
                                                                                                 │
                                                                                                END
    """
    # Crée un graphe d'état avec la structure RAGState
    graph = StateGraph(RAGState)

    # Enregistrement de tous les nœuds du graphe
    graph.add_node("load_pdf",          load_pdf)
    graph.add_node("chunk",             chunk)
    graph.add_node("extract_metadata",  extract_metadata)
    graph.add_node("embed_and_store",   embed_and_store)
    graph.add_node("rewrite_query",     rewrite_query)
    graph.add_node("expand_query",      expand_query)
    graph.add_node("hyde",              hyde)
    graph.add_node("self_query",        self_query)
    graph.add_node("retrieve",          retrieve)
    graph.add_node("rerank",            rerank)
    graph.add_node("generate",          generate)

    # Router — ingestion or query processing
    graph.add_conditional_edges(
        START,
        should_ingest,
        {"load_pdf": "load_pdf", "rewrite_query": "rewrite_query"},
    )

    # Ingestion pipeline
    graph.add_edge("load_pdf",          "chunk")
    graph.add_edge("chunk",             "extract_metadata")
    graph.add_edge("extract_metadata",  "embed_and_store")
    graph.add_edge("embed_and_store",   END)

    # Query processing pipeline
    graph.add_edge("rewrite_query",    "expand_query")
    graph.add_edge("expand_query",     "hyde")
    graph.add_edge("hyde",             "self_query")
    graph.add_edge("self_query",       "retrieve")
    graph.add_edge("retrieve",        "rerank")
    graph.add_edge("rerank",          "generate")
    graph.add_edge("generate",        END)

    compiled = graph.compile(checkpointer=checkpointer)
    log.info("Graph compiled (11 nodes)")
    return compiled
