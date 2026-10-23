"""Live monitor: reads decisions from Kafka and prints rolling stats.

This is one of the downstream services in the design: it never slows down
scoring, because it reads from Kafka in its own time. If it crashes and
restarts, it continues from where it stopped (Kafka remembers its position).

Run from the project root:
    python -m sentinel.monitor
"""

import argparse
import heapq
import json
import os
import time
from collections import Counter, deque

import numpy as np

from sentinel.events import DECISIONS_TOPIC, ensure_topic


class RollingStats:
    """Stats over the last `size` decisions, plus the riskiest payments seen."""

    def __init__(self, size: int = 1000, top_k: int = 5) -> None:
        self.recent: deque[dict] = deque(maxlen=size)  # old events fall out automatically
        self.total = 0
        self.top_k = top_k
        self.riskiest: list[tuple[float, str, dict]] = []  # min-heap of the top-k scores

    def add(self, event: dict) -> None:
        self.total += 1
        self.recent.append(event)
        item = (event["score"], event["txn_id"], event)
        if len(self.riskiest) < self.top_k:
            heapq.heappush(self.riskiest, item)
        elif item[:2] > self.riskiest[0][:2]:
            heapq.heapreplace(self.riskiest, item)  # drop the smallest of the top-k

    def summary(self) -> dict:
        decisions = Counter(e["decision"] for e in self.recent)
        n = max(len(self.recent), 1)
        latency = np.array([e["latency_ms"] for e in self.recent] or [0.0])
        return {
            "total": self.total,
            "window": len(self.recent),
            "share": {d: decisions[d] / n for d in ("approve", "review", "block")},
            "p50_ms": float(np.percentile(latency, 50)),
            "p99_ms": float(np.percentile(latency, 99)),
            "riskiest": [e for _, _, e in sorted(self.riskiest, reverse=True)],
        }


def render(summary: dict) -> str:
    share = summary["share"]
    lines = [
        f"Decisions: {summary['total']:,} total | last {summary['window']:,}: "
        f"approve {share['approve']:.1%}, review {share['review']:.1%}, block {share['block']:.1%}",
        f"API latency (last {summary['window']:,}): p50 {summary['p50_ms']:.1f} ms, p99 {summary['p99_ms']:.1f} ms",
        "Riskiest payments so far:",
    ]
    for e in summary["riskiest"]:
        lines.append(f"  {e['score']:.3f} {e['decision']:<7} card {e['card_id']} "
                     f"${e['amount']:,.2f} {e['category']} at {e['ts']} -> {', '.join(e['reasons'])}")
    return "\n".join(lines)


def main() -> None:
    from confluent_kafka import Consumer

    parser = argparse.ArgumentParser(description="Print live stats from the decisions topic.")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    parser.add_argument("--group", default="sentinel-monitor")
    parser.add_argument("--from-start", action="store_true", help="read the topic from the beginning")
    parser.add_argument("--every", type=float, default=3.0, help="seconds between printouts")
    args = parser.parse_args()

    consumer = Consumer({
        "bootstrap.servers": args.bootstrap,
        "group.id": args.group,
        "auto.offset.reset": "earliest" if args.from_start else "latest",
    })
    ensure_topic(args.bootstrap, DECISIONS_TOPIC)  # services may start before the API creates it
    consumer.subscribe([DECISIONS_TOPIC])
    stats, last_print = RollingStats(), time.monotonic()
    print(f"Listening to '{DECISIONS_TOPIC}' on {args.bootstrap} (Ctrl + C to stop)...")
    try:
        while True:
            msg = consumer.poll(0.5)
            if msg is not None and msg.error() is None:
                stats.add(json.loads(msg.value()))
            elif msg is not None:
                print(f"Kafka error: {msg.error()}")
            if stats.total and time.monotonic() - last_print >= args.every:
                print("\n" + render(stats.summary()))
                last_print = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()  # tells Kafka our position, so a restart continues from here


if __name__ == "__main__":
    main()
