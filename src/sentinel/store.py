"""Card memory: where each card's recent history lives.

Two stores with the SAME interface, so the API doesn't care which one it uses:
- InMemoryCardStore: a Python dict. Fast and simple, but lost on restart and
  not shared between API copies. Good for tests and local development.
- RedisCardStore: shared memory in Redis. Survives API restarts, and every
  API copy sees the same card history. This is what production uses.

Both return exactly the same clues as CardState (used in training), so the
model sees the same numbers live as it saw in training (no training-serving skew).
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from sentinel.features import WINDOWS, CardState, travel_speed_kmh

MAX_WINDOW_SECS = max(w.total_seconds() for w in WINDOWS.values())
KEY_TTL_SECS = 400 * 24 * 3600  # forget cards that are inactive for ~13 months


class CardStore(Protocol):
    """Anything that can give clues for a payment and then remember it."""

    def features_and_update(self, card_id: int, ts: datetime, amount: float,
                            lat: float = np.nan, lon: float = np.nan) -> dict: ...


class InMemoryCardStore:
    """Card memory in a Python dict (one per API process)."""

    def __init__(self) -> None:
        self.cards: defaultdict[int, CardState] = defaultdict(CardState)

    def features_and_update(self, card_id: int, ts: datetime, amount: float,
                            lat: float = np.nan, lon: float = np.nan) -> dict:
        card = self.cards[card_id]
        out = card.features(ts, amount, lat, lon)  # 1. clues from the past
        card.update(ts, amount, lat, lon)          # 2. then remember this payment
        return out

    def ping(self) -> bool:
        return True


# Runs INSIDE Redis, as one atomic step. Two requests for the same card can
# never interleave between "read history" and "add this payment".
#
# KEYS[1] = sorted set of recent payments (score = time, member = "seq:amount")
# KEYS[2] = hash with lifetime stats (n, total, last_ts, last_lat, last_lon, seq)
# ARGV    = now (epoch seconds), amount, max window (secs), key TTL (secs), lat, lon
_LUA = """
local recent_key, stats_key = KEYS[1], KEYS[2]
local now, amount = tonumber(ARGV[1]), ARGV[2]
local max_window, ttl = tonumber(ARGV[3]), tonumber(ARGV[4])

-- 1. Forget payments that are too old for every window (same rule as SlidingWindow: <=).
redis.call('ZREMRANGEBYSCORE', recent_key, '-inf', now - max_window)

-- 2. Read history BEFORE adding this payment (no peeking at the present).
local recent = redis.call('ZRANGE', recent_key, 0, -1, 'WITHSCORES')
local stats = redis.call('HMGET', stats_key, 'n', 'total', 'last_ts', 'last_lat', 'last_lon')

-- 3. Remember this payment.
local seq = redis.call('HINCRBY', stats_key, 'seq', 1)
redis.call('ZADD', recent_key, now, seq .. ':' .. amount)
redis.call('HINCRBY', stats_key, 'n', 1)
redis.call('HINCRBYFLOAT', stats_key, 'total', amount)
redis.call('HSET', stats_key, 'last_ts', ARGV[1], 'last_lat', ARGV[5], 'last_lon', ARGV[6])
redis.call('EXPIRE', recent_key, ttl)
redis.call('EXPIRE', stats_key, ttl)

return {recent, stats}
"""


def _text(value) -> str:
    """Redis may return bytes or str, depending on the client settings."""
    return value.decode() if isinstance(value, bytes) else value


def to_epoch(ts) -> float:
    """Seconds since 1970. Naive times are treated as UTC (same as pandas)."""
    return pd.Timestamp(ts).timestamp()


class RedisCardStore:
    """Card memory in Redis, shared by every API copy."""

    def __init__(self, client, prefix: str = "card") -> None:
        self.redis = client
        self.prefix = prefix
        self._script = client.register_script(_LUA)

    def _keys(self, card_id: int) -> list[str]:
        return [f"{self.prefix}:{card_id}:recent", f"{self.prefix}:{card_id}:stats"]

    def features_and_update(self, card_id: int, ts: datetime, amount: float,
                            lat: float = np.nan, lon: float = np.nan) -> dict:
        now = to_epoch(ts)
        recent, stats = self._script(
            keys=self._keys(card_id),
            args=[repr(now), repr(float(amount)), MAX_WINDOW_SECS, KEY_TTL_SECS,
                  repr(float(lat)), repr(float(lon))],
        )

        # recent = [member, score, member, score, ...]; member = "seq:amount"
        times = np.array([float(s) for s in recent[1::2]])
        amounts = np.array([float(_text(m).split(":")[1]) for m in recent[0::2]])

        out = {}
        for name, size in WINDOWS.items():
            in_window = times > now - size.total_seconds()
            out[f"count_{name}"] = int(in_window.sum())
            out[f"sum_{name}"] = float(amounts[in_window].sum())

        n, total, last_ts, last_lat, last_lon = (None if v is None else float(v) for v in stats)
        out["amount_vs_avg"] = amount / (total / n) if n else np.nan
        if last_ts is None:
            out["secs_since_last"] = np.nan
            out["speed_kmh"] = np.nan
        else:
            out["secs_since_last"] = now - last_ts
            out["speed_kmh"] = travel_speed_kmh(last_lat, last_lon, lat, lon, now - last_ts)
        return out

    def ping(self) -> bool:
        return bool(self.redis.ping())


def make_store() -> CardStore:
    """Use Redis when REDIS_URL is set, otherwise in-memory."""
    url = os.getenv("REDIS_URL")
    if not url:
        return InMemoryCardStore()
    import redis  # imported here so the project still runs without Redis installed

    return RedisCardStore(redis.Redis.from_url(url))
