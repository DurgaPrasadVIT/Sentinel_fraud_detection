"""Build clues (features) for every payment, using only the past.

Run from the project root:
    python -m sentinel.features
"""

from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.data import CATEGORY_COLUMNS, PROCESSED_DIR
from sentinel.window import SlidingWindow

FEATURES_DIR = Path("data/features")

WINDOWS = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}


class CardState:
    """Everything we remember about one card."""

    def __init__(self) -> None:
        self.windows = {name: SlidingWindow(size) for name, size in WINDOWS.items()}
        self.n_payments = 0
        self.total_spent = 0.0
        self.last_ts = None

    def features(self, ts, amount: float) -> dict:
        """Clues for a new payment, using only EARLIER payments."""
        out = {}
        for name, window in self.windows.items():
            count, total = window.snapshot(ts)
            out[f"count_{name}"] = count
            out[f"sum_{name}"] = total

        if self.n_payments:
            out["amount_vs_avg"] = amount / (self.total_spent / self.n_payments)
        else:
            out["amount_vs_avg"] = np.nan

        if self.last_ts is not None:
            out["secs_since_last"] = (ts - self.last_ts).total_seconds()
        else:
            out["secs_since_last"] = np.nan
        return out

    def update(self, ts, amount: float) -> None:
        """Remember this payment for future clues."""
        for window in self.windows.values():
            window.add(ts, amount)
        self.n_payments += 1
        self.total_spent += amount
        self.last_ts = ts


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two points on Earth."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_card_history(df: pd.DataFrame) -> pd.DataFrame:
    """Walk through payments in time order and add history clues per card."""
    states = defaultdict(CardState)
    rows = []
    for card, ts, amount in zip(df["card_id"], df["ts"], df["amount"]):
        state = states[card]
        rows.append(state.features(ts, amount))  # 1. clues from the past
        state.update(ts, amount)                 # 2. then remember this payment
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clues that only need the payment itself."""
    df = df.copy()
    df["hour"] = df["ts"].dt.hour.astype("int8")
    df["day_of_week"] = df["ts"].dt.day_of_week.astype("int8")
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 4)).astype("int8")
    df["distance_km"] = haversine_km(
        df["cust_lat"], df["cust_lon"], df["merch_lat"], df["merch_lon"]
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All clues, in time order."""
    df = df.sort_values("ts", kind="stable")
    return add_simple_features(add_card_history(df))


def main() -> None:
    """Build clues over the FULL timeline, then split back into train and test."""
    train = pd.read_parquet(PROCESSED_DIR / "fraudTrain.parquet").assign(split="train")
    test = pd.read_parquet(PROCESSED_DIR / "fraudTest.parquet").assign(split="test")
    full = pd.concat([train, test], ignore_index=True)
    for col in CATEGORY_COLUMNS:
        full[col] = full[col].astype("category")

    feats = build_features(full)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        part = feats[feats["split"] == split].drop(columns="split")
        out_path = FEATURES_DIR / f"{split}.parquet"
        part.to_parquet(out_path, index=False)
        print(f"Saved {out_path}: {len(part):,} rows, {part.shape[1]} columns")


if __name__ == "__main__":
    main()