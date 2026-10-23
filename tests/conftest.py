"""Test setup.

The real dataset and trained model are not in git. When the model is missing
(e.g. in CI), we train a tiny stand-in model on random data, so the API tests
still run and check the API's own logic: validation, card memory, events, rules.
The real model's quality is checked by the training and evaluation scripts.
"""

import os
import tempfile
from pathlib import Path

REAL_MODEL = Path("models/fraud_lgbm.txt")


def _build_tiny_model(root: Path) -> None:
    import lightgbm as lgb
    import numpy as np
    import pandas as pd

    from sentinel.api import CATEGORIES
    from sentinel.decisions import Policy
    from sentinel.train import FEATURES

    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame({name: rng.random(n) for name in FEATURES})
    df["amount"] = rng.uniform(1, 1000, n)
    df["category"] = pd.Categorical(rng.choice(CATEGORIES, n), categories=CATEGORIES)
    y = (df["amount"] > 600).astype(int)  # "fraud" = big amount: enough to give varied scores
    model = lgb.LGBMClassifier(n_estimators=20, num_leaves=7, verbose=-1).fit(df[FEATURES], y)

    (root / "models").mkdir()
    model.booster_.save_model(str(root / "models" / "fraud_lgbm.txt"))
    Policy(review_line=0.3, block_line=0.9).save(
        root / "models" / "policy.json", root / "models" / "fraud_lgbm.txt"
    )


def pytest_sessionstart(session) -> None:
    """Runs before tests are collected, so skip conditions see the model."""
    if REAL_MODEL.exists():
        return
    root = Path(tempfile.mkdtemp(prefix="sentinel-tests-"))
    _build_tiny_model(root)
    os.chdir(root)  # paths like models/ are relative to the working directory
    print(f"\nNo trained model found: using a tiny test model in {root}")
