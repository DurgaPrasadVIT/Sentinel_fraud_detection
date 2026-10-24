"""Tests for sentinel.casenotes and sentinel.cases."""

from sentinel.casenotes import (
    Facts,
    TemplateWriter,
    card_key,
    check_note,
    mask_card,
    risk_text,
    template_note,
    write_note,
)
from sentinel.cases import CaseService, CaseStore, prepare

EVENT = {
    "txn_id": "t1", "card_id": 4532015112830366, "ts": "2020-12-21T23:51:00",
    "amount": 850.0, "category": "shopping_net", "decision": "block", "score": 0.9731,
    "latency_ms": 4.2, "scored_at": "2026-10-23T21:00:00",
    "reasons": ["device_used_by_6_cards", "sum_24h", "amount_vs_avg", "is_night"],
    "clues": {"sum_24h": 1765.0, "amount_vs_avg": 12.34, "is_night": 1, "speed_kmh": None},
}


class FakeLLM:
    name = "fake"

    def __init__(self, text=None, fail=False):
        self.text, self.fail = text, fail

    def write(self, facts):
        if self.fail:
            raise TimeoutError("LLM timed out")
        return self.text


def test_facts_are_plain_english_and_masked():
    lines = "\n".join(Facts.from_event(EVENT).lines())
    assert "****0366" in lines and "4532015112830366" not in lines
    assert "device used by 6 different cards in 30 days" in lines
    assert "$1,765.00 spent on this card in the previous 24 hours" in lines
    assert "12.3x this card's average" in lines
    assert "made at night" in lines


def test_template_note_has_decision_reasons_and_next_step():
    note = template_note(Facts.from_event(EVENT))
    assert note.startswith("BLOCK (risk 97.3%)")
    assert "close and reissue" in note


def test_guard_accepts_numbers_from_the_facts_even_reformatted():
    facts = Facts.from_event(EVENT)
    note = ("Card ****0366 was blocked at 97% risk: $850 online at 23:51, after $1,765 "
            "in 24 hours and 12.3x its usual amount, on a device shared by 6 cards.")
    assert check_note(note, facts) == []


def test_guard_rejects_invented_numbers_and_full_card_numbers():
    facts = Facts.from_event(EVENT)
    assert any("4,000" in p or "4000" in p for p in check_note("Card spent $4,000 today.", facts))
    assert check_note("Card 4532015112830366 blocked.", facts)


def test_hallucinating_llm_falls_back_to_template():
    note = write_note(EVENT, FakeLLM("Blocked: the cardholder lost $9,999 in Paris."))
    assert note.source == "template" and note.rejected
    assert note.text.startswith("BLOCK")


def test_failing_llm_falls_back_to_template():
    note = write_note(EVENT, FakeLLM(fail=True))
    assert note.source == "template" and "timed out" in note.rejected[0]


def test_good_llm_note_is_kept():
    text = "Blocked at 97.3% risk: $850 shopping_net after $1,765 in 24 hours. Call the cardholder."
    note = write_note(EVENT, FakeLLM(text))
    assert note.source == "fake" and note.text == text


def test_case_service_saves_decisions_cases_and_rings():
    store = CaseStore(":memory:")
    service = CaseService(store, TemplateWriter())
    service.handle("decisions", {**EVENT, "txn_id": "ok", "decision": "approve", "reasons": []})
    service.handle("decisions", EVENT)
    service.handle("decisions", EVENT)  # Kafka re-delivery: must not duplicate
    service.handle("ring-alerts", {"cards": ["1111222233334444"], "devices": ["device:x"],
                                   "payments": 9, "amount": 900.0})
    service.flush(wait=True)
    db = store.db
    assert db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 2
    assert db.execute("SELECT card, note_source FROM cases").fetchall() == [("****0366", "template")]
    assert db.execute("SELECT cards FROM rings").fetchone()[0] == '["****4444"]'


def test_mask_card():
    assert mask_card(4532015112830366) == "****0366"


def test_notes_never_cite_who_the_cardholder_is():
    event = {**EVENT, "reasons": ["amount", "age", "city_pop"],
             "clues": {"amount": 850.0, "age": 61, "city_pop": 5000}}
    lines = "\n".join(Facts.from_event(event).lines())
    assert "age" not in lines.lower() and "5,000" not in lines


def test_risk_is_never_shown_as_100_or_0_percent():
    assert risk_text(0.99996) == ">99.9%"
    assert risk_text(0.9731) == "97.3%"
    assert risk_text(0.00002) == "<0.1%"
    assert "risk >99.9%" in template_note(Facts.from_event({**EVENT, "score": 0.99996}))


def test_card_key_groups_a_card_without_revealing_it():
    key = card_key(4532015112830366)
    assert key == card_key("4532015112830366") and len(key) == 16
    assert key != card_key(4532015112830367) and "0366" not in key


def test_note_mentions_earlier_flags_and_passes_the_guard():
    facts = Facts.from_event(EVENT, earlier=(2, 1464.99))
    note = template_note(facts)
    assert "2 other flagged payments on this card in the previous 24 hours ($1,464.99 in total)" in note
    assert check_note(note, facts) == []
    assert "Card history" not in template_note(Facts.from_event(EVENT))


def test_service_counts_only_earlier_flags_of_the_same_card_in_24h():
    service = CaseService(CaseStore(":memory:"), TemplateWriter())

    def flag(txn, card, ts, amount, decision="block"):
        e = {**EVENT, "txn_id": txn, "card_id": card, "ts": ts, "amount": amount, "decision": decision}
        return service.earlier_flags(e) if decision != "approve" else None

    assert flag("a", 1, "2020-06-23T01:00:00", 500.0) == (0, 0.0)
    flag("b", 2, "2020-06-24T02:00:00", 99.0)                     # another card
    assert flag("c", 1, "2020-06-24T02:00:00", 741.77) == (0, 0.0)  # "a" is >24h old
    assert flag("d", 1, "2020-06-24T03:33:40", 7.82) == (1, 741.77)
    assert flag("d", 1, "2020-06-24T03:33:40", 7.82) == (1, 741.77)  # redelivery: not counted twice
    assert flag("e", 1, "2020-06-24T04:00:00", 5.0) == (2, 749.59)


def test_old_database_gets_the_new_column_and_keeps_its_rows():
    import sqlite3

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE cases (txn_id TEXT PRIMARY KEY, ts TEXT, card TEXT, amount REAL, category TEXT, "
               "decision TEXT, score REAL, reasons TEXT, note TEXT, note_source TEXT, note_rejected TEXT, "
               "status TEXT DEFAULT 'open', created_at TEXT)")
    db.execute("INSERT INTO cases (txn_id) VALUES ('old')")
    prepare(db)
    prepare(db)  # running it again is harmless
    assert "card_key" in {r[1] for r in db.execute("PRAGMA table_info(cases)")}
    assert db.execute("SELECT txn_id, card_key FROM cases").fetchall() == [("old", None)]
