"""Tests for sentinel.api."""

import pytest
from fastapi.testclient import TestClient

from sentinel.api import app
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
        assert client.get("/health").json() == {"status": "ok"}


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