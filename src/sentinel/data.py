"""Load raw transaction data, clean it, and save it as Parquet.

Run from the project root:
    python -m sentinel.data
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Personal data (PII). Not needed to catch fraud, and risky to keep.
PII_COLUMNS = ["first", "last", "street", "job", "gender", "dob"]

# Short, clear names used everywhere else in the project.
RENAME = {
    "trans_date_trans_time": "ts",
    "cc_num": "card_id",
    "amt": "amount",
    "lat": "cust_lat",
    "long": "cust_lon",
    "merch_long": "merch_lon",
    "trans_num": "txn_id",
}

CATEGORY_COLUMNS = ["merchant", "category", "city", "state"]


def load_raw(path: Path) -> pd.DataFrame:
    """Read one raw CSV file exactly as downloaded."""
    return pd.read_csv(path, index_col=0, parse_dates=["trans_date_trans_time", "dob"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw data into clean data.

    Adds age, removes PII, renames columns, uses smaller types,
    and sorts by time.
    """
    df = df.copy()

    # Age at the time of payment. Age is a useful clue; birth date is PII.
    ts, dob = df["trans_date_trans_time"], df["dob"]
    birthday_passed = (ts.dt.month > dob.dt.month) | (
        (ts.dt.month == dob.dt.month) & (ts.dt.day >= dob.dt.day)
    )
    df["age"] = (ts.dt.year - dob.dt.year - (~birthday_passed)).astype("int16")

    df = df.drop(columns=PII_COLUMNS).rename(columns=RENAME)

    for col in CATEGORY_COLUMNS:
        df[col] = df[col].astype("category")
    df["is_fraud"] = df["is_fraud"].astype("int8")

    return df.sort_values("ts").reset_index(drop=True)


def build(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> None:
    """Clean both raw files and save them as Parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["fraudTrain", "fraudTest"]:
        df = clean(load_raw(raw_dir / f"{name}.csv"))
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved {out_path}: {len(df):,} rows, {df.shape[1]} columns")


if __name__ == "__main__":
    build()
