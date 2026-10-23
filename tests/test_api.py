"""Tests for sentinel.api."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

import sentinel.api
from sentinel.api import CATEGORIES, app
from sentinel.explain import load_model
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


def test_rejects_unknown_category():
    with TestClient(app) as client:
        response = client.post("/score", json={**PAYMENT, "category": "electronic"})
    assert response.status_code == 422


def test_rejects_impossible_location():
    with TestClient(app) as client:
        response = client.post("/score", json={**PAYMENT, "cust_lat": 200})
    assert response.status_code == 422


def test_mixed_timezones_for_same_card_do_not_crash():
    with TestClient(app) as client:
        first = client.post("/score", json={**PAYMENT, "card_id": 77})
        second = client.post(
            "/score", json={**PAYMENT, "card_id": 77, "ts": "2020-12-21T21:05:00Z"}
        )
    assert first.status_code == 200
    assert second.status_code == 200


def test_utc_and_other_timezones_mean_the_same_moment():
    """21:00 in New York (UTC-5 in December) is 02:00 UTC the next day."""
    from sentinel.api import Payment

    ny = Payment(**{**PAYMENT, "ts": "2020-12-21T21:00:00-05:00"})
    assert ny.ts.isoformat() == "2020-12-22T02:00:00"


def test_api_categories_match_the_model():
    """If the model is retrained with new categories, this reminds us to update the API."""
    assert CATEGORIES == load_model().pandas_categorical[0]


def test_every_decision_is_published_once(monkeypatch):
    published = []

    class ListPublisher:
        def publish(self, event):
            published.append(event)

        def close(self):
            pass

    monkeypatch.setattr(sentinel.api, "make_publisher", ListPublisher)
    with TestClient(app) as client:
        body = client.post("/score", json={**PAYMENT, "txn_id": "abc-1"}).json()
    assert body["txn_id"] == "abc-1"
    assert len(published) == 1
    assert published[0]["txn_id"] == "abc-1"
    assert published[0]["decision"] == body["decision"]
    assert published[0]["card_id"] == PAYMENT["card_id"]


def test_txn_id_is_created_when_missing():
    with TestClient(app) as client:
        body = client.post("/score", json=PAYMENT).json()
    assert len(body["txn_id"]) == 32
