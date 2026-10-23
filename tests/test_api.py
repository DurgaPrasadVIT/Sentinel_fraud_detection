"""Tests for sentinel.api."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

import sentinel.api
from sentinel.api import app
from sentinel.store import RedisCardStore
from sentinel.train import MODELS_DIR

# The trained model is not on GitHub (too big), so skip if it's missing.
pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "fraud_lgbm.txt").exists(), reason="trained model not found"
)

PAYMENT = {
    "card_id": 1,
    "ts": "2020-12-21T21:00:00",
    "amount": 25.0,
    "category": "gas_transport",
    "age": 35,
    "city_pop": 50000,
    "cust_lat": 40.72,
    "cust_lon": -74.04,
    "merch_lat": 40.75,
    "merch_lon": -74.0,
}


def test_health():
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body == {"status": "ok", "store": "InMemoryCardStore"}


def test_score_returns_a_valid_decision():
    with TestClient(app) as client:
        body = client.post("/score", json=PAYMENT).json()
    assert 0 <= body["score"] <= 1
    assert body["decision"] in {"approve", "review", "block"}
    assert body["latency_ms"] > 0


def test_approved_payments_have_no_reasons():
    with TestClient(app) as client:
        body = client.post("/score", json=PAYMENT).json()
    if body["decision"] == "approve":
        assert body["reasons"] == []


def test_rejects_negative_amount():
    with TestClient(app) as client:
        response = client.post("/score", json={**PAYMENT, "amount": -5})
    assert response.status_code == 422


def test_rejects_missing_card_id():
    bad = {k: v for k, v in PAYMENT.items() if k != "card_id"}
    with TestClient(app) as client:
        assert client.post("/score", json=bad).status_code == 422

def test_uses_redis_and_history_survives_restart(monkeypatch):
    server = fakeredis.FakeServer()
    monkeypatch.setattr(
        sentinel.api, "make_store", lambda: RedisCardStore(fakeredis.FakeRedis(server=server))
    )
    with TestClient(app) as client:  # first API process
        assert client.get("/health").json()["store"] == "RedisCardStore"
        client.post("/score", json={**PAYMENT, "card_id": 99})
    with TestClient(app) as client:  # restarted API process, same Redis
        client.post("/score", json={**PAYMENT, "card_id": 99, "ts": "2020-12-21T21:10:00"})
    stats = fakeredis.FakeRedis(server=server).hgetall("card:99:stats")
    assert stats[b"n"] == b"2"  # both payments remembered across the restart


def test_health_reports_redis_down(monkeypatch):
    class BrokenStore:
        def ping(self):
            raise ConnectionError("Redis is down")

    monkeypatch.setattr(sentinel.api, "make_store", BrokenStore)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
