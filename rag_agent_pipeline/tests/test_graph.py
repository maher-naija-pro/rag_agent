"""Tests for graph construction and routing logic."""

from langgraph.checkpoint.memory import InMemorySaver

from graph import should_ingest, build_graph


class TestRouter:
    def test_first_turn_routes_to_load_pdf(self, base_state):
        assert should_ingest(base_state(ingested=False)) == "load_pdf"

    def test_subsequent_turn_routes_to_cache_check(self, base_state):
        assert should_ingest(base_state(ingested=True)) == "cache_check"

    def test_missing_ingested_key_routes_to_load_pdf(self):
        state = {"messages": [], "question": "test"}
        assert should_ingest(state) == "load_pdf"


class TestBuildGraph:
    def test_compiles_without_error(self):
        graph = build_graph(InMemorySaver())
        assert graph is not None

    def test_has_all_nodes(self):
        graph = build_graph(InMemorySaver())
        node_names = set(graph.get_graph().nodes.keys())
        expected = {"load_pdf", "chunk", "extract_metadata", "embed_and_store", "cache_check", "cache_store", "rewrite_query", "expand_query", "hyde", "self_query", "retrieve", "rerank", "generate"}
        assert expected.issubset(node_names)
