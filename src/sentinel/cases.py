"""Case service: saves decisions, writes case notes, and records ring alerts.

Reads two Kafka topics:
- decisions    -> every decision is saved; review/block ones become CASES with a note
- ring-alerts  -> saved as ring cases

Everything goes into a small SQLite database (data/cases.db) that the dashboard
reads. In production this would be PostgreSQL; the code would barely change.

Run from the project root:
    python -m sentinel.cases
"""

import argparse
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sentinel.casenotes import Writer, make_writer, mask_card, write_note

CASES_DB = Path(os.getenv("SENTINEL_CASES_DB", "data/cases.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    txn_id TEXT PRIMARY KEY, ts TEXT, card TEXT, amount REAL, category TEXT,
    decision TEXT, score REAL, latency_ms REAL, scored_at TEXT
);
CREATE INDEX IF NOT EXISTS decisions_scored_at ON decisions(scored_at);
CREATE TABLE IF NOT EXISTS cases (
    txn_id TEXT PRIMARY KEY, ts TEXT, card TEXT, amount REAL, category TEXT,
    decision TEXT, score REAL, reasons TEXT, note TEXT, note_source TEXT,
    note_rejected TEXT, status TEXT DEFAULT 'open', created_at TEXT
);
CREATE TABLE IF NOT EXISTS rings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, cards TEXT, devices TEXT,
    payments INTEGER, amount REAL
);
"""


class CaseStore:
    """SQLite store. WAL mode lets the dashboard read while this service writes."""

    def __init__(self, path: Path | str = CASES_DB) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def add_decision(self, e: dict) -> None:
        # INSERT OR IGNORE: Kafka may deliver an event twice after a restart
        # ("at least once"). The txn_id primary key makes saving idempotent.
        self.db.execute(
            "INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?,?,?)",
            (e["txn_id"], str(e["ts"]), mask_card(e["card_id"]), e["amount"], e["category"],
             e["decision"], e["score"], e["latency_ms"], str(e.get("scored_at", ""))),
        )

    def add_case(self, e: dict, note) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO cases (txn_id, ts, card, amount, category, decision, score, "
            "reasons, note, note_source, note_rejected, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (e["txn_id"], str(e["ts"]), mask_card(e["card_id"]), e["amount"], e["category"],
             e["decision"], e["score"], json.dumps(e.get("reasons", [])), note.text, note.source,
             json.dumps(note.rejected), time.strftime("%Y-%m-%dT%H:%M:%S")),
        )

    def add_ring(self, alert: dict) -> None:
        self.db.execute(
            "INSERT INTO rings (created_at, cards, devices, payments, amount) VALUES (?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps([mask_card(c) for c in alert["cards"]]),
             json.dumps(alert["devices"]), alert["payments"], alert["amount"]),
        )

    def commit(self) -> None:
        self.db.commit()


class CaseService:
    """Turns Kafka messages into rows. LLM notes are written on a few background
    threads, so one slow note doesn't hold up thousands of decisions."""

    def __init__(self, store: CaseStore, writer: Writer, workers: int = 4) -> None:
        self.store, self.writer = store, writer
        self.pool = ThreadPoolExecutor(workers)
        self.pending = []

    def handle(self, topic: str, event: dict) -> None:
        if topic == "ring-alerts":
            self.store.add_ring(event)
            return
        self.store.add_decision(event)
        if event["decision"] != "approve":
            self.pending.append((event, self.pool.submit(write_note, event, self.writer)))

    def flush(self, wait: bool = False) -> int:
        """Save notes that are ready (all of them if wait=True). Returns how many."""
        done, still = [], []
        for event, future in self.pending:
            (done if wait or future.done() else still).append((event, future))
        for event, future in done:
            self.store.add_case(event, future.result())
        self.pending = still
        self.store.commit()
        return len(done)


def main() -> None:
    from confluent_kafka import Consumer

    from sentinel.events import DECISIONS_TOPIC, ensure_topic

    parser = argparse.ArgumentParser(description="Save decisions and write analyst case notes.")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    parser.add_argument("--from-start", action="store_true")
    args = parser.parse_args()

    writer = make_writer()
    store = CaseStore()
    service = CaseService(store, writer)
    # Create the topics we read before subscribing. Services start in any order (e.g. in
    # docker compose); a consumer that subscribes to a topic that doesn't exist yet only
    # notices it minutes later, and with "latest" it has missed everything sent meanwhile.
    ensure_topic(args.bootstrap, DECISIONS_TOPIC)
    ensure_topic(args.bootstrap, "ring-alerts", partitions=1)
    consumer = Consumer({
        "bootstrap.servers": args.bootstrap, "group.id": "sentinel-cases",
        "auto.offset.reset": "earliest" if args.from_start else "latest",
        "enable.auto.commit": False,  # commit our Kafka position only after rows are saved
    })
    consumer.subscribe([DECISIONS_TOPIC, "ring-alerts"])
    print(f"Case service: notes by '{writer.name}', saving to {CASES_DB} (Ctrl + C to stop)...")
    cases = 0
    try:
        while True:
            for msg in consumer.consume(num_messages=500, timeout=0.5):
                if msg.error() is None:
                    service.handle(msg.topic(), json.loads(msg.value()))
            saved = service.flush()
            if saved:
                cases += saved
                print(f"cases written: {cases}")
            if not service.pending:
                # Only now is everything read so far safely saved. After a crash,
                # Kafka re-delivers from here, and duplicates are ignored by txn_id.
                consumer.commit(asynchronous=True)
    except KeyboardInterrupt:
        pass
    finally:
        service.flush(wait=True)
        consumer.commit(asynchronous=False)
        consumer.close()


if __name__ == "__main__":
    main()
