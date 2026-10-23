"""Turn fraud scores into decisions (approve / review / block) using money.

Run from the project root:
    python -m sentinel.decisions
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.features import FEATURES_DIR
from sentinel.train import FEATURES, MODELS_DIR, TARGET, VALID_START, split_by_time

# Business assumptions. In a real bank, the fraud team provides these.
COST_FALSE_BLOCK = 20.0   # a good customer wrongly blocked
COST_REVIEW = 3.0         # analyst time per case
MAX_REVIEWS_PER_DAY = 50  # what the review team can handle

APPROVE, REVIEW, BLOCK = 0, 1, 2


@dataclass
class Policy:
    """Two lines that split scores into approve / review / block."""

    review_line: float
    block_line: float

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """Return APPROVE, REVIEW or BLOCK for each score."""
        return np.where(
            scores >= self.block_line, BLOCK,
            np.where(scores >= self.review_line, REVIEW, APPROVE),
        )


def money_lost(decisions: np.ndarray, is_fraud: np.ndarray, amount: np.ndarray) -> float:
    """Total cost of a set of decisions."""
    missed_fraud = amount[(decisions == APPROVE) & (is_fraud == 1)].sum()
    false_blocks = ((decisions == BLOCK) & (is_fraud == 0)).sum() * COST_FALSE_BLOCK
    reviews = (decisions == REVIEW).sum() * COST_REVIEW
    return float(missed_fraud + false_blocks + reviews)


def find_best_policy(scores, is_fraud, amount, n_days: int) -> Policy:
    """Try many line pairs; keep the cheapest one the review team can handle."""
    lines = np.linspace(0.01, 0.99, 99)
    best, best_cost = None, np.inf
    for review_line in lines:
        for block_line in lines[lines >= review_line]:
            policy = Policy(review_line, block_line)
            decisions = policy.decide(scores)
            if (decisions == REVIEW).sum() / n_days > MAX_REVIEWS_PER_DAY:
                continue  # too much work for the review team
            cost = money_lost(decisions, is_fraud, amount)
            if cost < best_cost:
                best, best_cost = policy, cost
    return best


def summarize(name: str, decisions, is_fraud, amount, n_days: int) -> None:
    """Print what a policy does, in business terms."""
    fraud_total = amount[is_fraud == 1].sum()
    stopped = amount[(decisions != APPROVE) & (is_fraud == 1)].sum()
    good_blocked = ((decisions == BLOCK) & (is_fraud == 0)).sum()
    print(f"\n{name}")
    print(f"  Approve {np.mean(decisions == APPROVE):.2%} | "
          f"Review {np.mean(decisions == REVIEW):.2%} | "
          f"Block {np.mean(decisions == BLOCK):.2%}")
    print(f"  Reviews per day: {(decisions == REVIEW).sum() / n_days:.0f}")
    print(f"  Fraud money stopped: {stopped / fraud_total:.1%} of ${fraud_total:,.0f}")
    print(f"  Good customers blocked: {good_blocked:,}")
    print(f"  Total money lost: ${money_lost(decisions, is_fraud, amount):,.0f}")


def main() -> None:
    booster = lgb.Booster(model_file=str(MODELS_DIR / "fraud_lgbm.txt"))
    _, valid = split_by_time(pd.read_parquet(FEATURES_DIR / "train.parquet"), VALID_START)
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")

    def prepare(df: pd.DataFrame):
        scores = booster.predict(df[FEATURES])
        n_days = df["ts"].dt.normalize().nunique()
        return scores, df[TARGET].to_numpy(), df["amount"].to_numpy(), n_days

    # 1. Choose the lines on VALID.
    v_scores, v_fraud, v_amount, v_days = prepare(valid)
    policy = find_best_policy(v_scores, v_fraud, v_amount, v_days)
    print(f"Best policy (chosen on valid): review >= {policy.review_line:.2f}, "
          f"block >= {policy.block_line:.2f}")

    # 2. Check it on TEST, and compare with simpler options.
    t_scores, t_fraud, t_amount, t_days = prepare(test)
    summarize("No model (approve everything)",
              np.full(len(t_scores), APPROVE), t_fraud, t_amount, t_days)
    summarize("Naive policy (block if score >= 0.5)",
              Policy(0.5, 0.5).decide(t_scores), t_fraud, t_amount, t_days)
    summarize("Money-based policy",
              policy.decide(t_scores), t_fraud, t_amount, t_days)


if __name__ == "__main__":
    main()