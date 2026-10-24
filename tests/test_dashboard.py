"""Tests for sentinel.dashboard (data functions and a smoke test of the page)."""

import pytest

from sentinel.casenotes import TemplateWriter
from sentinel.cases import CaseService, CaseStore
from sentinel.dashboard import (
    label_stats,
    other_cases,
    per_minute,
    review_queue,
    ring_alerts,
    set_status,
    summary,
)


def event(i, decision, score, amount=100.0):
    return {"txn_id": f"t{i}", "card_id": 4000000000000000 + i, "ts": f"2020-12-21T21:{i:02d}:00",
            "amount": amount, "category": "misc_net", "decision": decision, "score": score,
            "latency_ms": float(i), "scored_at": f"2026-10-23T21:{i:02d}:00", "reasons": ["amount"],
            "clues": {"amount": amount}}


@pytest.fixture
def db(tmp_path):
    store = CaseStore(tmp_path / "cases.db")
    service = CaseService(store, TemplateWriter())
    for i, (decision, score) in enumerate([("approve", 0.01), ("approve", 0.02),
                                            ("review", 0.4), ("block", 0.99)]):
        service.handle("decisions", event(i, decision, score))
    service.handle("ring-alerts", {"cards": ["1111", "2222"], "devices": ["device:a"],
                                   "payments": 6, "amount": 600.0})
    service.flush(wait=True)
    return store.db


def test_summary(db):
    s = summary(db)
    assert s["total"] == 4 and s["open_cases"] == 2 and s["rings"] == 1
    assert s["share"] == {"approve": 0.5, "review": 0.25, "block": 0.25}
    assert s["held_amount"] == 200.0


def test_review_queue_is_highest_risk_first(db):
    queue = review_queue(db)
    assert list(queue["decision"]) == ["block", "review"]
    assert queue.iloc[0]["note"].startswith("BLOCK")


def test_labels_move_cases_out_of_the_queue(db):
    set_status(db, "t3", "confirmed_fraud")
    assert list(review_queue(db)["txn_id"]) == ["t2"]
    assert label_stats(db) == {"Open": 1, "Confirmed fraud": 1}
    with pytest.raises(ValueError):
        set_status(db, "t2", "maybe")


def test_per_minute_and_rings(db):
    chart = per_minute(db)
    assert chart.sum().sum() == 4 and chart.index[0] == "21:00"
    ring = ring_alerts(db).iloc[0]
    assert ring["devices"] == 1 and ring["n_cards"] == 2


def test_dollar_signs_are_not_read_as_math():
    from sentinel.dashboard import plain

    assert plain("$13.63 then $2,093.49") == "\\$13.63 then \\$2,093.49"


def test_page_renders(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    monkeypatch.chdir(tmp_path)  # empty database: page must still load
    at = AppTest.from_file(str(__import__("sentinel.dashboard", fromlist=["x"]).__file__), default_timeout=30)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Sentinel: fraud operations"


def test_queue_appears_without_reloading_the_page(tmp_path, monkeypatch):
    """Cases created after the page loaded must show up on the next refresh."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import sentinel.dashboard as dash

    st.cache_resource.clear()  # the db connection is cached per process; start clean
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(dash.__file__, default_timeout=30)
    at.run()
    assert len(at.expander) == 0  # nothing yet

    store = CaseStore(tmp_path / "data" / "cases.db")
    service = CaseService(store, TemplateWriter())
    service.handle("decisions", event(1, "block", 0.99))
    service.flush(wait=True)

    at.run()  # what the 10 s timer triggers
    assert len(at.expander) == 1 and "BLOCK" in at.expander[0].label


def test_other_cases_on_the_same_card_include_later_ones(tmp_path):
    store = CaseStore(tmp_path / "cases.db")
    service = CaseService(store, TemplateWriter())
    card = 4111111111112620
    for txn, ts, amount in [("early", "2020-06-24T03:33:40", 7.82), ("late1", "2020-06-24T22:07:11", 723.23),
                            ("late2", "2020-06-24T23:15:28", 741.77)]:
        service.handle("decisions", {**event(0, "block", 0.99, amount), "txn_id": txn, "card_id": card, "ts": ts})
    service.handle("decisions", {**event(9, "block", 0.99), "card_id": 4999999999992620})  # same last 4 digits
    service.flush(wait=True)
    queue = review_queue(store.db).set_index("txn_id")
    others = other_cases(store.db, queue.loc["early", "card_key"], "early")
    assert list(others["amount"]) == [723.23, 741.77]  # later cases on this card; not the look-alike card
    assert other_cases(store.db, None, "early").empty
