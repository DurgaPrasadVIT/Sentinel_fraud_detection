"""Tests for sentinel.monitor."""

from sentinel.monitor import RollingStats, render


def event(i: int, score: float, decision: str = "approve") -> dict:
    return {"txn_id": f"t{i}", "card_id": i, "score": score, "decision": decision,
            "latency_ms": float(i), "amount": 10.0, "category": "home",
            "ts": "2020-12-21T21:00:00", "reasons": []}


def test_keeps_only_the_top_k_riskiest():
    stats = RollingStats(top_k=3)
    for i, score in enumerate([0.1, 0.9, 0.5, 0.99, 0.2, 0.7]):
        stats.add(event(i, score))
    assert [e["score"] for e in stats.summary()["riskiest"]] == [0.99, 0.9, 0.7]


def test_window_forgets_old_events():
    stats = RollingStats(size=2)
    stats.add(event(1, 0.1, "block"))
    stats.add(event(2, 0.1))
    stats.add(event(3, 0.1))
    summary = stats.summary()
    assert summary["total"] == 3 and summary["window"] == 2
    assert summary["share"]["block"] == 0  # the block event fell out of the window


def test_render_mentions_riskiest_card():
    stats = RollingStats()
    stats.add(event(7, 0.95, "block"))
    assert "card 7" in render(stats.summary())
