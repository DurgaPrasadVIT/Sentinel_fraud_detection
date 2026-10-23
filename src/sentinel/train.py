"""Train the fraud model with a time-based validation split.

Run from the project root:
    python -m sentinel.train
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sentinel.features import FEATURES_DIR

MODELS_DIR = Path("models")
VALID_START = "2020-04-01"
TARGET = "is_fraud"

# Clues the model may use. IDs and exact locations are left out on purpose:
# the model would just memorize specific cards, shops, and towns.
FEATURES = [
    "amount", "age", "city_pop", "category",
    "count_1h", "count_24h", "count_7d",
    "sum_1h", "sum_24h", "sum_7d",
    "amount_vs_avg", "secs_since_last", "speed_kmh",
    "hour", "day_of_week", "is_night", "distance_km",
]


def split_by_time(df: pd.DataFrame, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Everything before `start` is train, the rest is validation."""
    is_valid = df["ts"] >= start
    return df[~is_valid], df[is_valid]


def train_model(train: pd.DataFrame, valid: pd.DataFrame) -> lgb.LGBMClassifier:
    """Train LightGBM and stop when validation PR-AUC stops improving."""
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train[FEATURES],
        train[TARGET],
        eval_X=valid[FEATURES],
        eval_y=valid[TARGET],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def report(name: str, y_true: pd.Series, scores) -> None:
    """Print PR-AUC (main metric) and ROC-AUC."""
    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    print(f"{name:<24} PR-AUC {pr_auc:.3f} | ROC-AUC {roc_auc:.3f}")


def main() -> None:
    train_full = pd.read_parquet(FEATURES_DIR / "train.parquet")
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    train, valid = split_by_time(train_full, VALID_START)
    print(f"Train: {len(train):,} rows | Valid: {len(valid):,} rows | Test: {len(test):,} rows")

    model = train_model(train, valid)
    print(f"Best number of trees: {model.best_iteration_}")

    report("Baseline (amount only)", test[TARGET], test["amount"])
    report("Model on valid", valid[TARGET], model.predict_proba(valid[FEATURES])[:, 1])
    report("Model on test", test[TARGET], model.predict_proba(test[FEATURES])[:, 1])

    MODELS_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(MODELS_DIR / "fraud_lgbm.txt"))
    print(f"Saved model to {MODELS_DIR / 'fraud_lgbm.txt'}")


if __name__ == "__main__":
    main()