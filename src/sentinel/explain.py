"""Explain why the model gave a payment its score, using SHAP values.

Run from the project root:
    python -m sentinel.explain
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.decisions import Policy
from sentinel.features import FEATURES_DIR
from sentinel.scoring import top_reasons
from sentinel.train import FEATURES, MODELS_DIR


def load_model() -> lgb.Booster:
    """Load the saved model file (the same file the API will use)."""
    return lgb.Booster(model_file=str(MODELS_DIR / "fraud_lgbm.txt"))


def explain(booster: lgb.Booster, rows: pd.DataFrame, top_n: int = 3) -> list[list[tuple[str, float]]]:
    """For each row, the clues that pushed its score UP the most."""
    # One SHAP value per clue, plus a last column with the base value (we drop it).
    contrib = booster.predict(rows[FEATURES], pred_contrib=True)[:, :-1]
    return [top_reasons(row, top_n) for row in contrib]


def fmt(value) -> str:
    """Show numbers neatly."""
    return f"{value:,.2f}" if isinstance(value, float) else str(value)


def print_cases(title: str, booster: lgb.Booster, cases: pd.DataFrame) -> None:
    """Print each payment with its top reasons."""
    print(f"\n=== {title} ===")
    for (_, row), reasons in zip(cases.iterrows(), explain(booster, cases), strict=True):
        print(f"\nScore {row['score']:.3f} | fraud={row['is_fraud']} | "
              f"${row['amount']:,.2f} {row['category']} at {row['ts']:%Y-%m-%d %H:%M}")
        for name, push in reasons:
            print(f"  + {name} = {fmt(row[name])}  (push {push:+.2f})")


def main() -> None:
    booster = load_model()
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    test["score"] = booster.predict(test[FEATURES])

    # 1. Why were the riskiest payments flagged?
    print_cases("Top 3 riskiest payments", booster, test.nlargest(3, "score"))

    # 2. Why did we wrongly block some good customers?
    block_line = Policy.load().block_line
    false_blocks = test[(test["score"] >= block_line) & (test["is_fraud"] == 0)]
    print_cases("Good customers we wrongly blocked", booster, false_blocks.nlargest(3, "score"))

    # 3. Which clues matter most overall?
    sample = test.sample(20_000, random_state=42)
    contrib = booster.predict(sample[FEATURES], pred_contrib=True)[:, :-1]
    importance = pd.Series(np.abs(contrib).mean(axis=0), index=FEATURES)
    print("\n=== Most important clues overall (average push strength) ===")
    print(importance.sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
