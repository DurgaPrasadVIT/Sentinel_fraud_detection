"""Tests for sentinel.scoring.

The fast live path must give the SAME clues and the SAME score as the pandas
path used in training and offline evaluation.
"""

import numpy as np
import pandas as pd
import pytest

from sentinel.api import CATEGORIES
from sentinel.explain import explain, load_model
from sentinel.features import FEATURES_DIR
from sentinel.scoring import Scorer, top_reasons
from sentinel.train import FEATURES, MODELS_DIR

HISTORY = ["count_1h", "count_24h", "count_7d", "sum_1h", "sum_24h", "sum_7d",
           "amount_vs_avg", "secs_since_last", "speed_kmh"]

needs_model_and_data = pytest.mark.skipif(
    not (MODELS_DIR / "fraud_lgbm.txt").exists() or not (FEATURES_DIR / "test.parquet").exists(),
    reason="trained model or feature data not found",
)


def test_top_reasons_keeps_only_positive_pushes():
    contrib = np.zeros(len(FEATURES))
    contrib[FEATURES.index("amount")] = 2.0
    contrib[FEATURES.index("hour")] = 0.5
    contrib[FEATURES.index("age")] = -3.0
    assert top_reasons(contrib) == [("amount", 2.0), ("hour", 0.5)]


@pytest.fixture(scope="module")
def real_rows():
    df = pd.read_parquet(FEATURES_DIR / "test.parquet")
    # All fraud cases in a slice plus normal ones: covers both easy and risky payments.
    return pd.concat([df[df["is_fraud"] == 1].head(500), df.sample(2500, random_state=0)])


@needs_model_and_data
def test_fast_path_matches_pandas_path_on_real_payments(real_rows):
    booster = load_model()
    scorer = Scorer(booster, CATEGORIES)

    fast = np.vstack([
        scorer.vector(row, {k: row[k] for k in HISTORY})
        for row in real_rows.to_dict("records")
    ])
    slow = real_rows[FEATURES].copy()
    slow["category"] = slow["category"].cat.set_categories(CATEGORIES).cat.codes
    np.testing.assert_allclose(fast, slow.to_numpy(float), rtol=1e-12, equal_nan=True)

    np.testing.assert_allclose(
        booster.predict(fast, num_threads=1), booster.predict(real_rows[FEATURES]), rtol=1e-12
    )


@needs_model_and_data
def test_fast_reasons_match_offline_explanations(real_rows):
    booster = load_model()
    scorer = Scorer(booster, CATEGORIES)
    rows = real_rows[real_rows["is_fraud"] == 1].head(50)
    offline = [[name for name, _ in r] for r in explain(booster, rows)]
    live = [scorer.reasons(scorer.vector(row, {k: row[k] for k in HISTORY}))
            for row in rows.to_dict("records")]
    assert live == offline
