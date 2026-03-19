"""Tests for OpenAPI schema and CORS."""


class TestCORS:
    def test_cors_headers_present(self, client):
        r = client.options(
            "/api/health",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" in r.headers


class TestOpenAPI:
    def test_schema_loads(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200

    def test_all_routes_in_schema(self, client):
        paths = list(client.get("/openapi.json").json()["paths"].keys())
        assert "/api/ingest" in paths
        assert "/api/chat" in paths
        assert "/api/documents" in paths
        assert "/api/health" in paths
