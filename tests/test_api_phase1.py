from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "reports" / "demo_screen"


def test_contract_and_health(monkeypatch):
    monkeypatch.setenv("ACS_RESULTS_DIR", str(DEMO))
    client = TestClient(main.app)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["demo_mode"] is True

    r = client.get("/api/contract")
    assert r.status_code == 200
    assert "/api/components/{component_id}" in r.json()["resources"]["component"]


def test_summary_and_components(monkeypatch):
    monkeypatch.setenv("ACS_RESULTS_DIR", str(DEMO))
    client = TestClient(main.app)
    r = client.get("/api/summary")
    assert r.status_code == 200
    assert r.json()["total_components"] == 25

    r = client.get("/api/components?limit=3")
    assert r.status_code == 200
    assert len(r.json()["items"]) == 3
    assert {"part_id", "decision", "risk_score"}.issubset(r.json()["items"][0])


def test_component_packet(monkeypatch):
    monkeypatch.setenv("ACS_RESULTS_DIR", str(DEMO))
    client = TestClient(main.app)
    part_id = "DIGITAL_LOGIC_001_00001"
    r = client.get(f"/api/components/{part_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["component"]["part_id"] == part_id
    assert "forecast" in body
    assert "historical_trajectory" in body
    assert "anomaly_evidence" in body
    assert "decision" in body
    assert "explanation" in body
    assert body["forecast"]["horizon_h"] == 168.0


def test_missing_component(monkeypatch):
    monkeypatch.setenv("ACS_RESULTS_DIR", str(DEMO))
    client = TestClient(main.app)
    r = client.get("/api/components/NO_SUCH_PART")
    assert r.status_code == 404
