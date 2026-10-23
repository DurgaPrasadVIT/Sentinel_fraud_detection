"""Tests for sentinel.store.

The key test: Redis must give EXACTLY the same clues as CardState (used in
training). If they ever drift apart, the live model silently gets worse.
"""

import math
import random
from datetime import datetime, timedelta

import fakeredis
import pytest

from sentinel.store import KEY_TTL_SECS, InMemoryCardStore, RedisCardStore


def random_payments(n: int = 400, seed: int = 0):
    """Payments for a few cards, in time order, with bursts and long gaps."""
    rng = random.Random(seed)
    ts = datetime(2020, 1, 1)
    for _ in range(n):
        ts += timedelta(seconds=rng.choice([0, 30, 600, 3600, 3 * 86400, 9 * 86400]))
        yield rng.choice([1, 2, 3]), ts, round(rng.uniform(1, 1000), 2)


def same(a, b) -> bool:
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == pytest.approx(b, rel=1e-9, abs=1e-6)


@pytest.fixture
def redis_store():
    return RedisCardStore(fakeredis.FakeRedis())


def test_redis_matches_training_features(redis_store):
    memory = InMemoryCardStore()  # wraps CardState, the training logic
    for card, ts, amount in random_payments():
        expected = memory.features_and_update(card, ts, amount)
        got = redis_store.features_and_update(card, ts, amount)
        assert expected.keys() == got.keys()
        for key in expected:
            assert same(expected[key], got[key]), (key, expected[key], got[key])


def test_history_survives_api_restart():
    server = fakeredis.FakeServer()
    first_api = RedisCardStore(fakeredis.FakeRedis(server=server))
    first_api.features_and_update(7, datetime(2020, 1, 1, 10, 0), 50.0)

    restarted_api = RedisCardStore(fakeredis.FakeRedis(server=server))  # new process
    out = restarted_api.features_and_update(7, datetime(2020, 1, 1, 10, 30), 20.0)
    assert out["count_1h"] == 1
    assert out["sum_1h"] == 50.0


def test_cards_do_not_mix(redis_store):
    redis_store.features_and_update(1, datetime(2020, 1, 1, 10, 0), 50.0)
    out = redis_store.features_and_update(2, datetime(2020, 1, 1, 10, 5), 20.0)
    assert out["count_7d"] == 0


def test_keys_expire_so_redis_does_not_grow_forever(redis_store):
    redis_store.features_and_update(1, datetime(2020, 1, 1), 50.0)
    for key in redis_store._keys(1):
        assert 0 < redis_store.redis.ttl(key) <= KEY_TTL_SECS


def test_old_payments_are_removed_from_redis(redis_store):
    redis_store.features_and_update(1, datetime(2020, 1, 1), 50.0)
    redis_store.features_and_update(1, datetime(2020, 1, 20), 20.0)  # 19 days later
    recent_key = redis_store._keys(1)[0]
    assert redis_store.redis.zcard(recent_key) == 1  # only the new payment is kept
