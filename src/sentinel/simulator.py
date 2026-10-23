"""Payment simulator: plays the role of the card network.

Replays real payments from the test period (Jun-Dec 2020) into the scoring
API, as fast as you ask, and can inject fraud attacks. At the end it prints
what the system caught, because the simulator knows the true labels.

Run from the project root (API must be running):
    python -m sentinel.simulator --rate 50 --limit 3000 --attack
"""

import argparse
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from sentinel.data import PROCESSED_DIR

PAYMENT_FIELDS = [
    "txn_id", "card_id", "ts", "amount", "category", "age", "city_pop",
    "cust_lat", "cust_lon", "merch_lat", "merch_lon",
]


class TokenBucket:
    """Rate limiter: allows `rate` actions per second, with short bursts.

    The bucket holds up to `capacity` tokens and refills at `rate` per second.
    Each action takes one token; if the bucket is empty, wait until it refills.
    """

    def __init__(self, rate: float, capacity: float | None = None, clock=time.monotonic) -> None:
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate / 10)
        self.tokens = self.capacity
        self.clock = clock
        self.last = clock()

    def wait_time(self) -> float:
        """Seconds to wait before the next action (0 = go now). Takes a token if available."""
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return 0.0
        return (1 - self.tokens) / self.rate


def to_payment(row: dict) -> dict:
    """One dataset row -> the JSON body the API expects."""
    payment = {name: row[name] for name in PAYMENT_FIELDS}
    payment["ts"] = pd.Timestamp(row["ts"]).isoformat()
    payment["card_id"] = int(payment["card_id"])
    payment["age"] = int(payment["age"])
    payment["city_pop"] = int(payment["city_pop"])
    payment["category"] = str(payment["category"])
    for name in ["amount", "cust_lat", "cust_lon", "merch_lat", "merch_lon"]:
        payment[name] = float(payment[name])
    return payment


def card_testing_attack(victim: dict, rng: random.Random) -> list[dict]:
    """A thief with a stolen card: two big online purchases, then tiny 'is it still working?' tests."""
    start = pd.Timestamp(victim["ts"]) + timedelta(minutes=1)
    plan = [(0, 850.0, "shopping_net"), (3, 920.0, "misc_net")] + [
        (5 + 2 * i, round(rng.uniform(1, 15), 2), "gas_transport") for i in range(4)
    ]
    attack = []
    for i, (minutes, amount, category) in enumerate(plan):
        payment = {**victim, "txn_id": f"attack-{victim['txn_id']}-{i}"}
        payment.update(ts=(start + timedelta(minutes=minutes)).isoformat(),
                       amount=amount, category=category, is_fraud=1)
        attack.append(payment)
    return attack


def load_stream(limit: int, warmup_days: int, attack_every: int, seed: int = 0) -> list[dict]:
    """Payments in time order: optional warm-up period, then the measured part (+ attacks)."""
    df = pd.read_parquet(PROCESSED_DIR / "fraudTest.parquet").sort_values("ts", kind="stable")
    warm_end = df["ts"].min() + timedelta(days=warmup_days)
    warm = df[df["ts"] < warm_end]
    measured = df[df["ts"] >= warm_end].head(limit)

    rng = random.Random(seed)
    stream = [{**to_payment(r), "is_fraud": int(r["is_fraud"]), "warmup": True}
              for r in warm.to_dict("records")]
    for i, r in enumerate(measured.to_dict("records")):
        payment = {**to_payment(r), "is_fraud": int(r["is_fraud"]), "warmup": False}
        stream.append(payment)
        if attack_every and i and i % attack_every == 0:
            stream += [{**p, "warmup": False} for p in card_testing_attack(payment, rng)]
    # Keep everything in time order (attacks happen a few minutes after the victim's payment).
    stream.sort(key=lambda p: (not p["warmup"] and 1, p["ts"]))
    return stream


@dataclass
class Report:
    """What happened, from the card network's point of view."""

    decisions: Counter = field(default_factory=Counter)
    latencies: list = field(default_factory=list)
    fraud_seen: int = 0
    fraud_stopped: int = 0
    fraud_money: float = 0.0
    fraud_money_stopped: float = 0.0
    attacks_seen: int = 0
    attacks_stopped: int = 0
    errors: int = 0

    def add(self, payment: dict, body: dict | None, latency_ms: float) -> None:
        if body is None:
            self.errors += 1
            return
        self.decisions[body["decision"]] += 1
        self.latencies.append(latency_ms)
        stopped = body["decision"] != "approve"
        if payment["is_fraud"]:
            self.fraud_seen += 1
            self.fraud_stopped += stopped
            self.fraud_money += payment["amount"]
            self.fraud_money_stopped += payment["amount"] * stopped
        if str(payment["txn_id"]).startswith("attack-"):
            self.attacks_seen += 1
            self.attacks_stopped += stopped

    def print(self, seconds: float) -> None:
        total = sum(self.decisions.values())
        lat = np.array(self.latencies) if self.latencies else np.array([0.0])
        print(f"\nScored {total:,} payments in {seconds:.1f} s ({total / seconds:.0f}/s), errors: {self.errors}")
        print("Decisions: " + " | ".join(f"{k} {v:,} ({v / max(total, 1):.1%})"
                                         for k, v in self.decisions.most_common()))
        print(f"Round-trip latency: p50 {np.percentile(lat, 50):.1f} ms | "
              f"p99 {np.percentile(lat, 99):.1f} ms | max {lat.max():.1f} ms")
        if self.fraud_seen:
            print(f"Fraud payments stopped: {self.fraud_stopped}/{self.fraud_seen} "
                  f"(${self.fraud_money_stopped:,.0f} of ${self.fraud_money:,.0f})")
        if self.attacks_seen:
            print(f"Injected attack payments stopped: {self.attacks_stopped}/{self.attacks_seen}")


def send(client, url: str, payment: dict) -> tuple[dict | None, float]:
    body = {k: v for k, v in payment.items() if k not in {"is_fraud", "warmup"}}
    start = time.perf_counter()
    try:
        response = client.post(url, json=body, timeout=5)
        response.raise_for_status()
        return response.json(), (time.perf_counter() - start) * 1000
    except Exception:
        return None, (time.perf_counter() - start) * 1000


def send_all(client, url: str, payments: list[dict], workers: int,
             bucket: TokenBucket | None = None, report: Report | None = None) -> None:
    """Send payments with several workers. One card always uses the same worker,
    so its payments arrive in order and its history in Redis stays correct."""
    lanes = [[] for _ in range(workers)]
    for p in payments:
        lanes[p["card_id"] % workers].append(p)

    def lane_worker(lane: list[dict]) -> None:
        for p in lane:
            if bucket is not None:
                while (wait := bucket.wait_time()) > 0:
                    time.sleep(wait)
            body, latency = send(client, url, p)
            if report is not None:
                report.add(p, body, latency)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lane_worker, lanes))


def run(url: str, stream: list[dict], rate: float, workers: int) -> Report:
    import httpx2 as httpx

    warm = [p for p in stream if p["warmup"]]
    measured = [p for p in stream if not p["warmup"]]
    report = Report()
    with httpx.Client() as client:
        if warm:
            print(f"Warm-up: sending {len(warm):,} payments to build card history (not measured)...")
            send_all(client, url, warm, workers)
        print(f"Sending {len(measured):,} payments at up to {rate:.0f}/s...")
        start = time.perf_counter()
        send_all(client, url, measured, workers, TokenBucket(rate), report)
        report.print(time.perf_counter() - start)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay payments into the scoring API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/score")
    parser.add_argument("--rate", type=float, default=50, help="payments per second (measured part)")
    parser.add_argument("--limit", type=int, default=2000, help="payments to measure")
    parser.add_argument("--warmup-days", type=int, default=0,
                        help="replay this many days first to build card history (not measured)")
    parser.add_argument("--attack", action="store_true", help="inject a card-testing attack every 500 payments")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    stream = load_stream(args.limit, args.warmup_days, 500 if args.attack else 0)
    run(args.url, stream, args.rate, args.workers)


if __name__ == "__main__":
    main()
