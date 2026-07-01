"""
test_api.py — Unit + integration tests for the FastAPI backend.
Run with: pytest deployment/tests/ -v
"""

import pytest
from httpx import AsyncClient, ASGITransport
from deployment.api.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ── Health Check ─────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_health_check(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("healthy", "degraded")
    assert "version" in data


@pytest.mark.anyio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert "Hallucinations-in-AI" in response.json()["message"]


# ── Evaluate Endpoint ─────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_evaluate_missing_prompt(client):
    """Should fail if prompt is missing."""
    response = await client.post("/api/v1/evaluate/", json={})
    assert response.status_code == 422  # Validation error


@pytest.mark.anyio
async def test_evaluate_invalid_model(client):
    """Should fail with invalid model name."""
    response = await client.post(
        "/api/v1/evaluate/",
        json={"prompt": "Test prompt", "model": "invalid-model-xyz"},
    )
    assert response.status_code == 422


# ── Analyze Endpoint ──────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_analyze_missing_text(client):
    """Should return 422 if text is missing."""
    response = await client.post("/api/v1/analyze/", json={})
    assert response.status_code == 422


# ── Mitigate Endpoint ─────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_mitigate_missing_prompt(client):
    response = await client.post("/api/v1/mitigate/", json={})
    assert response.status_code == 422


# ── Results Endpoints ─────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_get_runs_empty(client):
    """Should return an empty list when no runs exist yet."""
    response = await client.get("/api/v1/results/runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.anyio
async def test_get_stats(client):
    response = await client.get("/api/v1/results/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_runs" in data
    assert "hallucination_rate_pct" in data


@pytest.mark.anyio
async def test_get_run_not_found(client):
    response = await client.get("/api/v1/results/runs/99999")
    assert response.status_code == 404
