"""Experiment tracking with MLflow.

Every training run records: settings, metrics, the money result, the model file,
and a fingerprint of the data it was trained on. So for any model in production
we can answer: which data, which code settings, how good was it, who replaced it.

Runs are stored locally in mlflow.db (+ mlruns/ for files). Browse them with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

EXPERIMENT = "sentinel-fraud"
RUN_INFO = Path("models/run.json")  # which MLflow run produced the current model


def setup():
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def data_fingerprint(df: pd.DataFrame) -> str:
    """Changes if any row, column or value changes. Logged with every run."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def save_run_info(run_id: str, model_version: str | int | None) -> None:
    RUN_INFO.write_text(json.dumps({"run_id": run_id, "model_version": model_version}, indent=2) + "\n")


def load_run_info() -> dict | None:
    return json.loads(RUN_INFO.read_text()) if RUN_INFO.exists() else None
