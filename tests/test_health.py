"""HTTP smoke tests for the Phase 1 FastAPI stub."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tooclose.server import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_endpoint_hides_nfiq_path() -> None:
    client = TestClient(app)
    response = client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert "anatomy" in body
    assert "nfiq_binary" not in str(body)
    assert body["camera"]["default_type"] in {"webcam", "phone"}
