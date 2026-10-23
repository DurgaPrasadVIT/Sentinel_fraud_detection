"""Tests for sentinel.explain."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from sentinel.explain import explain
from sentinel.train import FEATURES


def make_data(n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    """Fake data where fraud depends ONLY on the amount."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({name: rng.random(n) for name in FEATURES})
    df["category"] = pd.Categorical(rng.choice(["a", "b"], n))
    df["amount"] = rng.uniform(0, 1000, n)
    y = (df["amount"] > 500).astype(int)
    return df, y


def test_top_reason_is_the_clue_that_causes_fraud():
    df, y = make_data()
    model = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(df[FEATURES], y)
    big_payment = df[df["amount"] > 900].head(1)
    reasons = explain(model.booster_, big_payment)
    assert reasons[0][0][0] == "amount"


def test_only_returns_reasons_that_push_up():
    df, y = make_data()
    model = lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(df[FEATURES], y)
    reasons = explain(model.booster_, df.head(10))
    for row_reasons in reasons:
        for _, push in row_reasons:
            assert push > 0