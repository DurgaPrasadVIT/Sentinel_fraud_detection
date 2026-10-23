"""Model-ops service: shadow scoring and drift monitoring on live traffic.

Reads the decisions topic (like the case service), so it never slows down or
breaks payments.

SHADOW: every payment is also scored by the challenger model. Its decision is
recorded next to the champion's but NEVER used. This is how a new model is
tested on real traffic with zero risk; only after it proves itself would it go
to an A/B test (a small share of real decisions) and then full rollout.

DRIFT: the last `window` payments are compared with the training snapshot
(models/reference.json) every `check_every` payments, using PSI per clue and
for the champion's scores.

    python -m sentinel.modelops
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import pandas as pd

from sentinel.cases import CASES_DB, CaseStore
from sentinel.decisions import LABELS_BY_CODE, Policy
from sentinel.drift import compare, load_reference
from sentinel.train import FEATURES, MODELS_DIR

REFERENCE_PATH = MODELS_DIR / "reference.json"


class ModelOps:
    def __init__(self, store: CaseStore, challenger=None, challenger_policy: Policy | None = None,
                 reference: dict | None = None, window: int = 5000, check_every: int = 1000) -> None:
        self.store = store
        self.challenger, self.challenger_policy = challenger, challenger_policy
        self.reference = reference
        self.recent: deque[dict] = deque(maxlen=window)  # old events fall out automatically
        self.check_every = check_every
        self.since_check = 0
        self.last_report: pd.DataFrame | None = None

    def handle(self, events: list[dict]) -> None:
        """Process a batch of decision events (scoring a batch is much faster than one by one)."""
        events = [e for e in events if e.get("clues")]
        if not events:
            return
        if self.challenger is not None:
            self._shadow(events)
        self.recent.extend(events)
        self.since_check += len(events)
        if self.reference is not None and self.since_check >= self.check_every:
            self.check_drift()

    def _shadow(self, events: list[dict]) -> None:
        clues = pd.DataFrame([e["clues"] for e in events]).reindex(columns=FEATURES)
        scores = self.challenger.score(clues)
        decisions = self.challenger_policy.decide(scores)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.store.db.executemany(
            "INSERT OR IGNORE INTO shadow VALUES (?,?,?,?,?,?)",
            [(e["txn_id"], e["score"], e["decision"], float(s), LABELS_BY_CODE[int(d)], now)
             for e, s, d in zip(events, scores, decisions, strict=True)],
        )

    def check_drift(self) -> pd.DataFrame:
        self.since_check = 0
        live = pd.DataFrame([e["clues"] for e in self.recent]).reindex(columns=FEATURES)
        times = pd.to_datetime([e["ts"] for e in self.recent], format="ISO8601", utc=True)
        span_hours = (times.max() - times.min()).total_seconds() / 3600
        report = compare(self.reference, live, np.array([e["score"] for e in self.recent]), span_hours)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.store.db.execute("DELETE FROM drift")  # keep only the latest check
        self.store.db.executemany(
            "INSERT INTO drift VALUES (?,?,?,?,?)",
            [(now, len(self.recent), r.feature, float(r.psi), r.status) for r in report.itertuples()],
        )
        self.last_report = report
        return report


def load_challenger():
    """Challenger model + its policy, if they've been trained (python -m sentinel.challenger)."""
    from sentinel.challenger import CHALLENGER_PATH, CHALLENGER_POLICY_PATH, ChallengerScorer

    if not CHALLENGER_PATH.exists():
        return None, None
    return ChallengerScorer.load(), Policy.load(CHALLENGER_POLICY_PATH, CHALLENGER_PATH)


def main() -> None:
    from confluent_kafka import Consumer

    from sentinel.events import DECISIONS_TOPIC, ensure_topic

    parser = argparse.ArgumentParser(description="Shadow scoring and drift monitoring.")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    parser.add_argument("--window", type=int, default=5000)
    parser.add_argument("--check-every", type=int, default=1000)
    args = parser.parse_args()

    challenger, policy = load_challenger()
    reference = load_reference(REFERENCE_PATH) if REFERENCE_PATH.exists() else None
    ops = ModelOps(CaseStore(CASES_DB), challenger, policy, reference, args.window, args.check_every)
    print(f"Model-ops: shadow={'on' if challenger else 'off (no challenger)'}, "
          f"drift={'on' if reference else 'off (no reference)'} (Ctrl + C to stop)...")

    ensure_topic(args.bootstrap, DECISIONS_TOPIC)
    consumer = Consumer({"bootstrap.servers": args.bootstrap, "group.id": "sentinel-modelops",
                         "auto.offset.reset": "latest", "enable.auto.commit": False})
    consumer.subscribe([DECISIONS_TOPIC])
    try:
        while True:
            msgs = [m for m in consumer.consume(num_messages=500, timeout=1.0) if m.error() is None]
            if not msgs:
                continue
            ops.handle([json.loads(m.value()) for m in msgs])
            ops.store.commit()
            consumer.commit(asynchronous=True)
            if ops.last_report is not None and ops.since_check == 0:
                flagged = ops.last_report[ops.last_report["status"].isin(["watch", "alert"])]
                print(f"\nDrift check on last {len(ops.recent):,} payments: "
                      + ("all clues stable" if flagged.empty else
                         ", ".join(f"{r.feature} PSI {r.psi:.2f} ({r.status})" for r in flagged.itertuples())))
    except KeyboardInterrupt:
        pass
    finally:
        ops.store.commit()
        consumer.close()


if __name__ == "__main__":
    main()
