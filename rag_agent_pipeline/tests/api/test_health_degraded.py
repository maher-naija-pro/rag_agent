"""Tests for /api/health degraded path (real Qdrant client, bad endpoint)."""

from qdrant_client import QdrantClient


class TestHealthDegraded:
    def test_qdrant_unreachable_returns_degraded(self, client, monkeypatch):
        """When Qdrant client points to a bad endpoint, health returns degraded."""
        bad_client = QdrantClient(url="http://localhost:1", timeout=1)

        import config.qdrant as qdrant_mod
        # Replace the singleton so get_client() returns the bad client
        monkeypatch.setattr(qdrant_mod, "_client", bad_client)

        import config as config_mod
        monkeypatch.setattr(config_mod, "get_client", lambda: bad_client)

        r = client.get("/api/health")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "degraded"
        assert "error" in data["qdrant"]
