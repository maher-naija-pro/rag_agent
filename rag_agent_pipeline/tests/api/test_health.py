"""Tests for GET /api/health."""


class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_has_status_field(self, client):
        data = client.get("/api/health").json()
        assert "status" in data

    def test_has_sessions_count(self, client):
        data = client.get("/api/health").json()
        assert data["sessions"] == 0
