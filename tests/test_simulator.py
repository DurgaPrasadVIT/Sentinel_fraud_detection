"""Tests for sentinel.simulator."""

import random

import pandas as pd

from sentinel.simulator import Report, TokenBucket, card_testing_attack, to_payment

ROW = {
    "txn_id": "abc", "card_id": 4242, "ts": pd.Timestamp("2020-12-21 21:00"), "amount": 25.0,
    "category": "gas_transport", "age": 35, "city_pop": 50000, "cust_lat": 40.72,
    "cust_lon": -74.04, "merch_lat": 40.75, "merch_lon": -74.0, "merchant": "shop", "is_fraud": 0,
}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_token_bucket_allows_burst_then_waits():
    clock = FakeClock()
    bucket = TokenBucket(rate=10, capacity=2, clock=clock)
    assert bucket.wait_time() == 0
    assert bucket.wait_time() == 0
    assert bucket.wait_time() > 0  # bucket empty: must wait


def test_token_bucket_refills_over_time():
    clock = FakeClock()
    bucket = TokenBucket(rate=10, capacity=1, clock=clock)
    bucket.wait_time()
    clock.now = 0.1  # 0.1 s at 10/s = one new token
    assert bucket.wait_time() == 0


def test_to_payment_has_only_api_fields_with_plain_types():
    payment = to_payment(ROW)
    assert "merchant" not in payment and "is_fraud" not in payment
    assert payment["ts"] == "2020-12-21T21:00:00"
    assert isinstance(payment["card_id"], int)


def test_card_testing_attack_shape():
    attack = card_testing_attack(to_payment(ROW), random.Random(0))
    assert all(p["card_id"] == 4242 and p["is_fraud"] == 1 for p in attack)
    assert attack[0]["amount"] > 500                    # big purchase first
    assert all(p["amount"] < 20 for p in attack[2:])    # then tiny "test" payments
    times = [pd.Timestamp(p["ts"]) for p in attack]
    assert times == sorted(times) and times[-1] - times[0] < pd.Timedelta(minutes=30)


def test_report_counts_fraud_stopped():
    report = Report()
    report.add({**ROW, "is_fraud": 1, "txn_id": "attack-1"}, {"decision": "block"}, 10.0)
    report.add({**ROW, "is_fraud": 1}, {"decision": "approve"}, 12.0)
    report.add({**ROW}, None, 5000.0)  # an error
    assert report.fraud_seen == 2 and report.fraud_stopped == 1
    assert report.attacks_stopped == 1 and report.errors == 1
