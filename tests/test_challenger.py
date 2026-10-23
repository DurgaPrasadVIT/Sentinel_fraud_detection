"""Tests for sentinel.challenger and sentinel.tracking."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from sentinel.challenger import ChallengerScorer
from sentinel.tracking import data_fingerprint
from sentinel.train import FEATURES

CATS = ["grocery_pos", "shopping_net"]


def normal_traffic(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({name: rng.normal(0, 1, n) for name in FEATURES})
    df["category"] = rng.choice(CATS, n)
    return df


def fitted():
    train = normal_traffic()
    scorer = ChallengerScorer(IsolationForest(n_estimators=50, random_state=0), CATS, np.array([0.0]))
    scorer.forest.fit(scorer.matrix(train))
    scorer.calibration = np.sort(scorer.anomaly(normal_traffic(seed=1)))
    return scorer


def test_unusual_payments_score_higher_without_any_labels():
    scorer = fitted()
    normal = normal_traffic(200, seed=2)
    weird = normal.copy()
    weird["amount"] = 50.0  # 50 standard deviations away
    # Clearly higher, but not ~1.0: each tree splits on random clues, so one extreme clue
    # out of 17 only partly isolates a payment (one reason this model is weak at fraud).
    assert scorer.score(weird).mean() > scorer.score(normal).mean() + 0.2


def test_missing_history_and_unknown_category_do_not_crash():
    row = normal_traffic(1)
    row.loc[0, "amount_vs_avg"] = np.nan
    row.loc[0, "category"] = "travel"
    assert 0 <= fitted().score(row)[0] <= 1


def test_save_and_load(tmp_path):
    scorer = fitted()
    scorer.save(tmp_path / "c.joblib")
    again = ChallengerScorer.load(tmp_path / "c.joblib")
    data = normal_traffic(50, seed=3)
    np.testing.assert_array_equal(scorer.score(data), again.score(data))


def test_data_fingerprint_detects_any_change():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert data_fingerprint(df) == data_fingerprint(df.copy())
    changed = df.copy()
    changed.loc[2, "a"] = 4
    assert data_fingerprint(df) != data_fingerprint(changed)
