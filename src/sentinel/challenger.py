"""Challenger model: Isolation Forest (unsupervised anomaly detection).

The champion (LightGBM) learned from labelled fraud. An Isolation Forest needs
no labels at all: it learns what normal payments look like and scores how
"isolated" (unusual) a payment is. Why consider it?
- labels arrive late (chargebacks take weeks) and new fraud patterns have none yet;
- it can flag strange behaviour the champion never saw in training.

It is evaluated exactly like the champion (same test period, same money metric)
and then run in SHADOW mode on live traffic: scored, compared, never acted on.

    python -m sentinel.challenger
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from sentinel.decisions import POLICY_PATH, Policy, find_best_policy, money_lost, summarize
from sentinel.features import FEATURES_DIR
from sentinel.train import FEATURES, MODELS_DIR, TARGET, VALID_START, split_by_time

CHALLENGER_PATH = MODELS_DIR / "challenger_iforest.joblib"
CHALLENGER_POLICY_PATH = MODELS_DIR / "challenger_policy.json"
FIT_ROWS = 200_000  # an Isolation Forest only needs a sample to learn "normal"


class ChallengerScorer:
    """Turns clues into a 0-1 risk score: the share of validation payments that were LESS
    unusual than this one. Policies and comparisons then work like for the champion."""

    def __init__(self, forest: IsolationForest, categories: list[str], calibration: np.ndarray) -> None:
        self.forest = forest
        self.category_code = {name: i for i, name in enumerate(categories)}
        self.calibration = np.sort(calibration)

    def matrix(self, clues: pd.DataFrame) -> np.ndarray:
        x = clues[FEATURES].copy()
        x["category"] = x["category"].astype(str).map(self.category_code).fillna(-1)
        return x.astype(float).fillna(-1.0).to_numpy()  # -1 = "no history yet"

    def anomaly(self, clues: pd.DataFrame) -> np.ndarray:
        return -self.forest.score_samples(self.matrix(clues))  # higher = more unusual

    def score(self, clues: pd.DataFrame) -> np.ndarray:
        raw = self.anomaly(clues)
        return np.searchsorted(self.calibration, raw, side="right") / len(self.calibration)

    def save(self, path=CHALLENGER_PATH) -> None:
        joblib.dump({"forest": self.forest, "categories": list(self.category_code),
                     "calibration": self.calibration}, path)

    @classmethod
    def load(cls, path=CHALLENGER_PATH) -> "ChallengerScorer":
        d = joblib.load(path)
        return cls(d["forest"], d["categories"], d["calibration"])


def main() -> None:
    from sentinel.api import CATEGORIES
    from sentinel.tracking import setup

    train_full = pd.read_parquet(FEATURES_DIR / "train.parquet")
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    train, valid = split_by_time(train_full, VALID_START)
    days = lambda df: df["ts"].dt.normalize().nunique()  # noqa: E731

    mlflow = setup()
    with mlflow.start_run(run_name="isolation-forest-challenger") as run:
        mlflow.set_tag("role", "challenger")
        params = {"n_estimators": 200, "max_samples": 256, "random_state": 42, "fit_rows": FIT_ROWS}
        mlflow.log_params({**params, "features": ",".join(FEATURES), "uses_labels": False})

        sample = train.sample(min(FIT_ROWS, len(train)), random_state=42)
        forest = IsolationForest(n_estimators=params["n_estimators"], max_samples=params["max_samples"],
                                 random_state=params["random_state"], n_jobs=-1)
        scorer = ChallengerScorer(forest, CATEGORIES, np.array([0.0]))
        forest.fit(scorer.matrix(sample))  # no labels used
        scorer.calibration = np.sort(scorer.anomaly(valid))

        v_scores, t_scores = scorer.score(valid), scorer.score(test)
        policy = find_best_policy(v_scores, valid[TARGET].to_numpy(), valid["amount"].to_numpy(), days(valid))
        print(f"Challenger policy (chosen on valid): review >= {policy.review_line:.2f}, "
              f"block >= {policy.block_line:.2f}")

        champion = Policy.load()
        import lightgbm as lgb
        champ_scores = lgb.Booster(model_file=str(MODELS_DIR / "fraud_lgbm.txt")).predict(test[FEATURES])
        y, amount = test[TARGET].to_numpy(), test["amount"].to_numpy()
        rows = {}
        for name, scores, pol in [("Champion (LightGBM)", champ_scores, champion),
                                  ("Challenger (Isolation Forest)", t_scores, policy)]:
            result = summarize(name, pol.decide(scores), y, amount, days(test))
            rows[name] = {"PR-AUC": average_precision_score(y, scores),
                          "money lost": money_lost(pol.decide(scores), y, amount), **result}
        print("\n" + pd.DataFrame(rows).T[["PR-AUC", "money lost", "fraud_money_stopped_share",
                                           "good_customers_blocked"]].round(4).to_string())

        ch = rows["Challenger (Isolation Forest)"]
        mlflow.log_metrics({"test_pr_auc": ch["PR-AUC"], "test_money_lost": ch["money lost"],
                            "test_fraud_money_stopped_share": ch["fraud_money_stopped_share"],
                            "test_good_customers_blocked": ch["good_customers_blocked"]})
        scorer.save()
        policy.save(CHALLENGER_POLICY_PATH, CHALLENGER_PATH)
        mlflow.log_artifact(str(CHALLENGER_PATH))
        mlflow.log_artifact(str(CHALLENGER_POLICY_PATH))
        print(f"\nSaved {CHALLENGER_PATH} | MLflow run {run.info.run_id[:8]} (not registered for "
              f"production: it is a shadow candidate). Champion policy file: {POLICY_PATH}")


if __name__ == "__main__":
    main()
