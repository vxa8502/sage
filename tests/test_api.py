"""Tests for sage.api.routes — API endpoint behavior.

Uses a test app with mocked state to avoid loading heavy models.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sage.api.routes import router
from sage.core.models import ProductScore, RetrievedChunk


def _make_app(**state_overrides) -> FastAPI:
    """Create a test app with mocked state."""
    app = FastAPI()
    app.include_router(router)

    # Mock Qdrant client
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])

    # Mock cache
    mock_cache = MagicMock()
    mock_cache.get.return_value = (None, "miss")
    mock_cache.stats.return_value = SimpleNamespace(
        size=0,
        max_entries=100,
        exact_hits=0,
        semantic_hits=0,
        misses=0,
        evictions=0,
        hit_rate=0.0,
        ttl_seconds=3600.0,
        similarity_threshold=0.92,
        avg_semantic_similarity=0.0,
    )

    # Mock explainer with client attribute for health check
    mock_explainer = MagicMock()
    mock_explainer.client = MagicMock()

    app.state.qdrant = state_overrides.get("qdrant", mock_qdrant)
    app.state.embedder = state_overrides.get("embedder", MagicMock())
    app.state.detector = state_overrides.get("detector", MagicMock())
    app.state.explainer = state_overrides.get("explainer", mock_explainer)
    app.state.cache = state_overrides.get("cache", mock_cache)

    return app


@pytest.fixture
def client():
    """Test client with default mocked state."""
    app = _make_app()
    return TestClient(app)


@pytest.fixture
def sample_product() -> ProductScore:
    """Sample product for recommendation tests."""
    return ProductScore(
        product_id="P1",
        score=0.9,
        chunk_count=2,
        avg_rating=4.5,
        evidence=[
            RetrievedChunk(
                text="Good", score=0.9, product_id="P1", rating=4.5, review_id="r1"
            ),
        ],
    )


class TestHealthEndpoint:
    @patch("sage.api.routes.collection_exists", return_value=True)
    def test_healthy_when_all_components_available(self, mock_collection_exists):
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["qdrant_connected"] is True
            assert data["llm_reachable"] is True

    @patch("sage.api.routes.collection_exists", return_value=True)
    def test_degraded_when_qdrant_available_but_llm_unavailable(
        self, mock_collection_exists
    ):
        app = _make_app(explainer=None)
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["qdrant_connected"] is True
            assert data["llm_reachable"] is False

    @patch("sage.api.routes.collection_exists", return_value=False)
    def test_unhealthy_when_qdrant_unavailable(self, mock_collection_exists):
        app = _make_app()
        with TestClient(app) as c:
            resp = c.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "unhealthy"
            assert data["qdrant_connected"] is False


class TestRecommendEndpoint:
    def test_missing_query_returns_422(self, client):
        # POST with empty body should fail validation
        resp = client.post("/recommend", json={})
        assert resp.status_code == 422

    @patch("sage.api.routes.get_candidates", return_value=[])
    def test_empty_results(self, mock_get_candidates, client):
        resp = client.post("/recommend", json={"query": "test query", "explain": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] == []

    @patch("sage.api.routes.get_candidates")
    def test_returns_products_without_explain(
        self, mock_get_candidates, sample_product
    ):
        mock_get_candidates.return_value = [sample_product]
        app = _make_app()
        with TestClient(app) as c:
            resp = c.post("/recommend", json={"query": "headphones", "explain": False})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["recommendations"]) == 1
            rec = data["recommendations"][0]
            assert rec["product_id"] == "P1"
            assert rec["rank"] == 1
            # Response uses 'score' not 'relevance_score' (killer demo format)
            assert "score" in rec
            assert "explanation" not in rec or rec["explanation"] is None

    @patch("sage.api.routes.get_candidates")
    def test_request_with_filters(self, mock_get_candidates, sample_product):
        mock_get_candidates.return_value = [sample_product]
        app = _make_app()
        with TestClient(app) as c:
            resp = c.post(
                "/recommend",
                json={
                    "query": "laptop for video editing",
                    "k": 5,
                    "filters": {"min_rating": 4.5, "max_price": 1500},
                    "explain": False,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["recommendations"]) == 1

    @patch("sage.api.routes.get_candidates")
    def test_explainer_unavailable_returns_503(
        self, mock_get_candidates, sample_product
    ):
        mock_get_candidates.return_value = [sample_product]
        mock_embedder = MagicMock()
        mock_embedder.embed_single_query.return_value = [0.1] * 384
        app = _make_app(explainer=None, embedder=mock_embedder)
        with TestClient(app) as c:
            resp = c.post("/recommend", json={"query": "headphones", "explain": True})
            assert resp.status_code == 503
            assert "unavailable" in resp.json()["error"].lower()


class TestCacheEndpoints:
    def test_cache_stats(self, client):
        resp = client.get("/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "size" in data
        assert "hit_rate" in data

    def test_cache_clear(self, client):
        resp = client.post("/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
