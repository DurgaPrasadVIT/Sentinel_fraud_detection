"""Tests for sentinel.modelops (shadow scoring and live drift checks)."""

import numpy as np

from sentinel.cases import CaseStore
from sentinel.dashboard import drift_report, shadow_summary
from sentinel.decisions import Policy
from sentinel.drift import build_reference
from sentinel.modelops import ModelOps
from sentinel.train import FEATURES


class FakeChallenger:
    """Scores = amount / 1000, so big payments look risky."""

    def score(self, clues):
        return np.clip(clues["amount"].to_numpy(float) / 1000, 0, 1)


def event(i, amount, decision="approve", score=0.01, ts=None):
    clues = {name: 1.0 for name in FEATURES}
    clues.update(amount=amount, category="grocery_pos")
    return {"txn_id": f"t{i}", "card_id": i, "ts": ts or f"2020-12-21T{i % 24:02d}:00:00",
            "amount": amount, "decision": decision, "score": score, "clues": clues}


def reference_from(events):
    import pandas as pd

    df = pd.DataFrame([e["clues"] for e in events])
    df["category"] = df["category"].astype("category")
    return build_reference(df, FEATURES, np.array([e["score"] for e in events]))


def test_shadow_records_both_decisions_but_never_changes_events():
    store = CaseStore(":memory:")
    ops = ModelOps(store, FakeChallenger(), Policy(0.5, 0.9), reference=None)
    events = [event(1, 100.0), event(2, 950.0)]
    before = [dict(e) for e in events]
    ops.handle(events)
    assert events == before  # the live decision is untouched
    rows = store.db.execute("SELECT txn_id, champion_decision, challenger_decision FROM shadow").fetchall()
    assert rows == [("t1", "approve", "approve"), ("t2", "approve", "block")]
    s = shadow_summary(store.db)
    assert s["n"] == 2 and s["agreement"] == 0.5 and s["only_challenger"] == 1


def test_drift_check_runs_every_n_events_and_flags_bigger_amounts():
    rng = np.random.default_rng(0)
    normal = [event(i, float(rng.lognormal(3.5, 1))) for i in range(2000)]
    store = CaseStore(":memory:")
    ops = ModelOps(store, reference=reference_from(normal), window=1000, check_every=1000)
    ops.handle(normal[:999])
    assert ops.last_report is None                       # not yet
    ops.handle([event(i, float(rng.lognormal(3.5, 1)) * 3) for i in range(1000)])
    report = drift_report(store.db).set_index("feature")
    assert report.loc["amount", "status"] == "alert"


def test_events_without_clues_are_ignored():
    store = CaseStore(":memory:")
    ops = ModelOps(store, FakeChallenger(), Policy(0.5, 0.9))
    ops.handle([{"txn_id": "x", "decision": "approve", "score": 0.1}])
    assert shadow_summary(store.db) == {"n": 0}


def test_drift_check_accepts_mixed_timestamp_formats():
    # Real events mix "...T10:00:00" and "...T10:00:00.294529"; this used to crash the check.
    rng = np.random.default_rng(1)
    normal = [event(i, float(rng.lognormal(3.5, 1))) for i in range(200)]
    live = [event(i, float(rng.lognormal(3.5, 1)), ts=f"2020-12-21T10:{i % 60:02d}:00" + (".294529" if i % 2 else ""))
            for i in range(100)]
    store = CaseStore(":memory:")
    ops = ModelOps(store, reference=reference_from(normal), window=100, check_every=100)
    ops.handle(live)
    assert ops.last_report is not None
