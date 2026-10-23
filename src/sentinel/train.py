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


PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def train_model(train: pd.DataFrame, valid: pd.DataFrame) -> lgb.LGBMClassifier:
    """Train LightGBM and stop when validation PR-AUC stops improving."""
    model = lgb.LGBMClassifier(**PARAMS, verbose=-1)
    model.fit(
        train[FEATURES],
        train[TARGET],
        eval_X=valid[FEATURES],
        eval_y=valid[TARGET],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def report(name: str, y_true: pd.Series, scores) -> dict:
    """Print PR-AUC (main metric) and ROC-AUC, and return them."""
    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    print(f"{name:<24} PR-AUC {pr_auc:.3f} | ROC-AUC {roc_auc:.3f}")
    return {"pr_auc": pr_auc, "roc_auc": roc_auc}


def main() -> None:
    from sentinel.drift import build_reference, save_reference
    from sentinel.tracking import data_fingerprint, save_run_info, setup

    train_full = pd.read_parquet(FEATURES_DIR / "train.parquet")
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    train, valid = split_by_time(train_full, VALID_START)
    print(f"Train: {len(train):,} rows | Valid: {len(valid):,} rows | Test: {len(test):,} rows")

    mlflow = setup()
    with mlflow.start_run(run_name="lightgbm-champion") as run:
        mlflow.set_tag("role", "champion")
        mlflow.log_params({**PARAMS, "features": ",".join(FEATURES), "valid_start": VALID_START,
                           "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test),
                           "train_data": data_fingerprint(train[["txn_id", "ts", "amount", TARGET]])})

        model = train_model(train, valid)
        print(f"Best number of trees: {model.best_iteration_}")

        metrics = {"best_iteration": model.best_iteration_}
        for name, df, scores in [
            ("baseline_test", test, test["amount"]),
            ("valid", valid, model.predict_proba(valid[FEATURES])[:, 1]),
            ("test", test, model.predict_proba(test[FEATURES])[:, 1]),
        ]:
            label = {"baseline_test": "Baseline (amount only)", "valid": "Model on valid",
                     "test": "Model on test"}[name]
            for metric, value in report(label, df[TARGET], scores).items():
                metrics[f"{name}_{metric}"] = value
        mlflow.log_metrics(metrics)

        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "fraud_lgbm.txt"
        model.booster_.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        # What "normal" looked like in training: drift checks compare live traffic to this.
        valid_scores = model.predict_proba(valid[FEATURES])[:, 1]
        save_reference(build_reference(valid, FEATURES, valid_scores), MODELS_DIR / "reference.json")
        mlflow.log_artifact(str(MODELS_DIR / "reference.json"))

        info = mlflow.lightgbm.log_model(model.booster_, name="model",
                                         registered_model_name="sentinel-fraud")
        version = getattr(info, "registered_model_version", None)
        save_run_info(run.info.run_id, version)
        print(f"Saved model to {model_path} | MLflow run {run.info.run_id[:8]}, "
              f"registered as sentinel-fraud v{version}")


if __name__ == "__main__":
    main()
