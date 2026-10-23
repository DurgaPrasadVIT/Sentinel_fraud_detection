"""Live fraud operations dashboard.

Reads the case database written by `python -m sentinel.cases` and refreshes
every few seconds. Analysts can mark cases as fraud / not fraud: those answers
are the labels a bank uses to retrain the model.

Run from the project root:
    streamlit run src/sentinel/dashboard.py
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from sentinel.cases import CASES_DB, SCHEMA

STATUSES = {"open": "Open", "confirmed_fraud": "Confirmed fraud", "not_fraud": "Not fraud"}


def connect(path: Path | str = CASES_DB) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # fresh machine: no data/ folder yet
    db = sqlite3.connect(str(path), check_same_thread=False, timeout=5)
    db.executescript(SCHEMA)  # works even before the case service has started
    return db


# ---------- Data (plain functions, tested without a browser) ----------

def summary(db: sqlite3.Connection, last_n: int = 2000) -> dict:
    recent = pd.read_sql(
        "SELECT decision, amount, latency_ms FROM decisions ORDER BY scored_at DESC LIMIT ?",
        db, params=(last_n,))
    total = db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    open_cases = db.execute("SELECT COUNT(*) FROM cases WHERE status = 'open'").fetchone()[0]
    rings = db.execute("SELECT COUNT(*) FROM rings").fetchone()[0]
    n = max(len(recent), 1)
    share = recent["decision"].value_counts().reindex(["approve", "review", "block"], fill_value=0) / n
    held = recent.loc[recent["decision"] != "approve", "amount"].sum()
    return {
        "total": int(total), "window": len(recent), "open_cases": int(open_cases),
        "rings": int(rings), "share": share.to_dict(), "held_amount": float(held),
        "p50_ms": float(recent["latency_ms"].quantile(0.5)) if len(recent) else 0.0,
        "p99_ms": float(recent["latency_ms"].quantile(0.99)) if len(recent) else 0.0,
    }


def per_minute(db: sqlite3.Connection, minutes: int = 30) -> pd.DataFrame:
    """Decisions per minute (wall-clock time when scored), last `minutes` minutes."""
    df = pd.read_sql("SELECT decision, scored_at FROM decisions ORDER BY scored_at DESC LIMIT 200000", db)
    if df.empty:
        return pd.DataFrame(columns=["approve", "review", "block"])
    df["minute"] = pd.to_datetime(df["scored_at"], format="ISO8601").dt.floor("min")
    df = df[df["minute"] >= df["minute"].max() - pd.Timedelta(minutes=minutes)]
    table = df.pivot_table(index="minute", columns="decision", aggfunc="size", fill_value=0)
    return table.reindex(columns=["approve", "review", "block"], fill_value=0)


def review_queue(db: sqlite3.Connection, status: str = "open", limit: int = 50) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT txn_id, ts, card, amount, category, decision, score, reasons, note, note_source "
        "FROM cases WHERE status = ? ORDER BY score DESC, ts DESC LIMIT ?",
        db, params=(status, limit))
    df["reasons"] = df["reasons"].map(lambda r: ", ".join(json.loads(r)))
    return df


def ring_alerts(db: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql("SELECT created_at, cards, devices, payments, amount FROM rings ORDER BY id DESC", db)
    df["cards"] = df["cards"].map(lambda c: ", ".join(json.loads(c)))
    df["devices"] = df["devices"].map(lambda d: len(json.loads(d)))
    return df


def label_stats(db: sqlite3.Connection) -> dict:
    rows = db.execute("SELECT status, COUNT(*) FROM cases GROUP BY status").fetchall()
    return {STATUSES.get(s, s): n for s, n in rows}


def set_status(db: sqlite3.Connection, txn_id: str, status: str) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    db.execute("UPDATE cases SET status = ? WHERE txn_id = ?", (status, txn_id))
    db.commit()


# ---------- Page ----------

def app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Sentinel fraud operations", layout="wide")
    st.title("Sentinel: fraud operations")
    db = st.cache_resource(connect)()

    @st.fragment(run_every="3s")
    def live() -> None:
        s = summary(db)
        cols = st.columns(6)
        cols[0].metric("Payments scored", f"{s['total']:,}")
        cols[1].metric("Blocked (last 2k)", f"{s['share']['block']:.2%}")
        cols[2].metric("Sent to review", f"{s['share']['review']:.2%}")
        cols[3].metric("Money held for checks", f"${s['held_amount']:,.0f}")
        cols[4].metric("API latency p99", f"{s['p99_ms']:.1f} ms", help=f"p50 {s['p50_ms']:.1f} ms")
        cols[5].metric("Open cases / ring alerts", f"{s['open_cases']} / {s['rings']}")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Decisions per minute")
            chart = per_minute(db)
            if chart.empty:
                st.caption("Waiting for decisions...")
            else:
                st.bar_chart(chart, color=["#9ca3af", "#f59e0b", "#dc2626"])
        with right:
            st.subheader("Ring alerts")
            rings = ring_alerts(db)
            if rings.empty:
                st.caption("No rings detected yet.")
            else:
                st.dataframe(rings, hide_index=True, use_container_width=True)
            st.caption(f"Analyst labels so far: {label_stats(db) or 'none'}")

    live()

    st.subheader("Review queue (highest risk first)")
    queue = review_queue(db)
    if queue.empty:
        st.caption("No open cases. Start the simulator and the case service.")
    for case in queue.head(15).itertuples():
        title = (f"{case.decision.upper()}  |  risk {case.score:.1%}  |  ${case.amount:,.2f} "
                 f"{case.category}  |  card {case.card}  |  {case.ts}")
        with st.expander(title):
            st.write(case.note)
            st.caption(f"Model reasons: {case.reasons}  |  note by: {case.note_source}")
            a, b, _ = st.columns([1, 1, 4])
            if a.button("Confirm fraud", key=f"f-{case.txn_id}"):
                set_status(db, case.txn_id, "confirmed_fraud")
                st.rerun()
            if b.button("Not fraud", key=f"n-{case.txn_id}"):
                set_status(db, case.txn_id, "not_fraud")
                st.rerun()


if __name__ == "__main__":
    app()
