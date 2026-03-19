"""Tests for /api/health degraded path."""

from unittest.mock import patch, MagicMock


class TestHealthDegraded:
    def test_qdrant_unreachable_returns_degraded(self, client):
        """When Qdrant client raises, health returns degraded (not 500)."""
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = ConnectionError("refused")

        with patch("config.get_client", return_value=mock_client):
            r = client.get("/api/health")

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "degraded"
        assert "error" in data["qdrant"]
