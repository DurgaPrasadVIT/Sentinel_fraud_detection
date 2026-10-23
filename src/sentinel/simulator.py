"""Payment simulator: plays the role of the card network.

Replays real payments from the test period (Jun-Dec 2020) into the scoring
API, as fast as you ask, and can inject fraud attacks. At the end it prints
what the system caught, because the simulator knows the true labels.

Run from the project root (API must be running):
    python -m sentinel.simulator --rate 50 --limit 3000 --attack
    python -m sentinel.simulator --rate 50 --limit 3000 --warmup-days 3 --attack --rings 5
"""

import argparse
import hashlib
import random
import time
from collections import Counter, defaultdict
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


def device_id(*parts) -> str:
    """A stable, anonymous device fingerprint."""
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


class Devices:
    """Gives every card realistic devices.

    - Most people: 1 phone; some: a phone and a laptop.
    - About 3% of cards belong to households of 2-3 cards that share a family tablet.
      That's normal behaviour the ring detector must NOT flag.
    """

    def __init__(self, card_ids, seed: int = 0, household_share: float = 0.03) -> None:
        rng = random.Random(seed)
        self.own = {c: [device_id("own", c, k) for k in range(1 if rng.random() < 0.7 else 2)]
                    for c in card_ids}
        shared = rng.sample(sorted(card_ids), int(len(card_ids) * household_share))
        self.family_tablet = {}
        while len(shared) >= 2:
            size = 3 if len(shared) >= 3 and rng.random() < 0.3 else 2
            group, shared = shared[:size], shared[size:]
            tablet = device_id("household", *group)
            self.family_tablet.update({c: tablet for c in group})
        self.rng = rng

    def pick(self, card_id) -> str:
        if card_id in self.family_tablet and self.rng.random() < 0.2:
            return self.family_tablet[card_id]
        return self.rng.choice(self.own[card_id])


def mule_ring(ring_no: int, cards: list[int], start: pd.Timestamp, profiles: dict,
              rng: random.Random) -> list[dict]:
    """A money-mule ring: several stolen cards cashed out from 2 shared devices.

    Payments are deliberately small and spread over hours ("low and slow"), so
    each one looks normal on its own. Only the shared devices give it away.
    """
    devices = [device_id("mule", ring_no, k) for k in range(2)]
    payments = []
    for i, card in enumerate(cards):
        for j in range(3):
            ts = start + timedelta(minutes=rng.randint(0, 12 * 60))
            # Card i uses device i % 2, and the first card uses both: one connected group.
            device = devices[(i + j) % 2] if i == 0 else devices[i % 2]
            payments.append({
                **profiles[card], "txn_id": f"ring{ring_no}-{card}-{j}", "card_id": card,
                "device_id": device, "ts": ts.isoformat(), "amount": round(rng.uniform(40, 250), 2),
                "category": rng.choice(["misc_net", "shopping_net", "grocery_pos"]),
                "is_fraud": 1, "ring": ring_no,
            })
    return payments


def card_testing_attack(victim: dict, rng: random.Random) -> list[dict]:
    """A thief with a stolen card: two big online purchases, then tiny 'is it still working?' tests."""
    start = pd.Timestamp(victim["ts"]) + timedelta(minutes=1)
    plan = [(0, 850.0, "shopping_net"), (3, 920.0, "misc_net")] + [
        (5 + 2 * i, round(rng.uniform(1, 15), 2), "gas_transport") for i in range(4)
    ]
    attack = []
    for i, (minutes, amount, category) in enumerate(plan):
        payment = {**victim, "txn_id": f"attack-{victim['txn_id']}-{i}"}  # same device as the victim's payment
        payment.update(ts=(start + timedelta(minutes=minutes)).isoformat(),
                       amount=amount, category=category, is_fraud=1)
        attack.append(payment)
    return attack


def load_stream(limit: int, warmup_days: int, attack_every: int, n_rings: int = 0,
                seed: int = 0) -> list[dict]:
    """Payments in time order: optional warm-up period, then the measured part
    (+ card-testing attacks and mule rings)."""
    df = pd.read_parquet(PROCESSED_DIR / "fraudTest.parquet").sort_values("ts", kind="stable")
    warm_end = df["ts"].min() + timedelta(days=warmup_days)
    warm = df[df["ts"] < warm_end]
    measured = df[df["ts"] >= warm_end].head(limit)

    rng = random.Random(seed)
    devices = Devices(df["card_id"].unique(), seed)

    def payment(row: dict, warmup: bool) -> dict:
        p = to_payment(row)
        return {**p, "device_id": devices.pick(p["card_id"]), "is_fraud": int(row["is_fraud"]),
                "warmup": warmup}

    stream = [payment(r, True) for r in warm.to_dict("records")]
    for i, r in enumerate(measured.to_dict("records")):
        p = payment(r, False)
        stream.append(p)
        if attack_every and i and i % attack_every == 0:
            stream += [{**a, "warmup": False} for a in card_testing_attack(p, rng)]

    if n_rings:
        # Stolen cards keep their owner's details (age, home); only shop and device change.
        profiles = {p["card_id"]: p for p in stream if not p["warmup"]}
        cards = sorted(profiles)
        t0, t1 = measured["ts"].min(), measured["ts"].max()
        for ring_no in range(n_rings):
            members = rng.sample(cards, rng.randint(6, 9))
            start = t0 + (t1 - t0) * rng.random() * 0.5
            stream += [{**p, "warmup": False}
                       for p in mule_ring(ring_no, members, start, profiles, rng)]

    # Keep everything in time order.
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
    ring_seen: int = 0
    ring_stopped: int = 0
    errors: int = 0
    events: list = field(default_factory=list)  # payment + decision, for ring detection

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
        if "ring" in payment:
            self.ring_seen += 1
            self.ring_stopped += stopped
        self.events.append({**payment, "decision": body["decision"]})

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
        if self.ring_seen:
            print(f"Injected mule-ring payments stopped: {self.ring_stopped}/{self.ring_seen}")
            self.print_ring_detection()

    def print_ring_detection(self) -> None:
        """Run the graph detector over everything that happened, and score it."""
        from sentinel.rings import RingDetector

        detector = RingDetector()
        frozen: set[str] = set()  # cards of rings already alerted: the bank would freeze them
        saved = ring_money = 0.0
        for event in sorted(self.events, key=lambda e: e["ts"]):
            if "ring" in event:
                ring_money += event["amount"]
                if str(event["card_id"]) in frozen:
                    saved += event["amount"]
            ring = detector.add(event)
            if ring:
                frozen |= ring.cards
        truth = defaultdict(set)
        for e in self.events:
            if "ring" in e:
                truth[e["ring"]].add(str(e["card_id"]))
        ring_cards = set().union(*truth.values())
        found = detector.rings()
        flagged = set().union(*(r.cards for r in found)) if found else set()
        caught = sum(1 for cards in truth.values() if cards <= flagged)
        print(f"Ring detector: {len(found)} groups flagged | rings fully found: {caught}/{len(truth)} | "
              f"ring cards flagged: {len(flagged & ring_cards)}/{len(ring_cards)} | "
              f"innocent cards flagged: {len(flagged - ring_cards)}")
        if ring_money:
            print(f"Ring money after the alert (stoppable by freezing the ring's cards): "
                  f"${saved:,.0f} of ${ring_money:,.0f} ({saved / ring_money:.0%})")


def send(client, url: str, payment: dict) -> tuple[dict | None, float]:
    body = {k: v for k, v in payment.items() if k not in {"is_fraud", "warmup", "ring"}}
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
    parser.add_argument("--rings", type=int, default=0, help="inject this many money-mule rings")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    stream = load_stream(args.limit, args.warmup_days, 500 if args.attack else 0, args.rings)
    run(args.url, stream, args.rate, args.workers)


if __name__ == "__main__":
    main()
