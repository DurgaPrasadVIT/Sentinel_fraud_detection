"""Tests for sentinel.drift."""

import numpy as np
import pandas as pd

from sentinel.drift import build_reference, compare, psi


def make(n=5000, seed=0, amount_scale=1.0, online_share=0.2):
    rng = np.random.default_rng(seed)
    cats = np.where(rng.random(n) < online_share, "shopping_net", "grocery_pos")
    return pd.DataFrame({
        "amount": rng.lognormal(3.5, 1.0, n) * amount_scale,
        "amount_vs_avg": np.where(rng.random(n) < 0.05, np.nan, rng.lognormal(0, 0.5, n)),
        "category": pd.Categorical(cats, categories=["grocery_pos", "shopping_net"]),
    })


FEATURES = ["amount", "amount_vs_avg", "category"]


def test_psi_is_zero_for_identical_and_grows_with_shift():
    a = {"x": 0.5, "y": 0.5}
    assert psi(a, a) == 0
    assert psi(a, {"x": 0.6, "y": 0.4}) < psi(a, {"x": 0.9, "y": 0.1})


def test_same_kind_of_traffic_is_not_drift():
    train = make(seed=0)
    ref = build_reference(train, FEATURES, np.random.default_rng(0).random(5000))
    live = make(seed=1)  # new sample, same behaviour
    report = compare(ref, live, np.random.default_rng(1).random(5000))
    assert (report["status"] == "ok").all(), report


def test_bigger_amounts_and_more_online_shopping_are_flagged():
    train = make(seed=0)
    ref = build_reference(train, FEATURES, np.random.default_rng(0).random(5000))
    live = make(seed=1, amount_scale=3.0, online_share=0.6)
    report = compare(ref, live).set_index("feature")
    assert report.loc["amount", "status"] == "alert"
    assert report.loc["category", "status"] == "alert"
    assert report.loc["amount_vs_avg", "status"] == "ok"  # untouched clue stays quiet


def test_unknown_category_and_missing_values_have_their_own_buckets():
    train = make(seed=0)
    ref = build_reference(train, FEATURES, np.zeros(5000))
    live = pd.DataFrame({"category": ["travel"] * 100, "amount": [None] * 100, "amount_vs_avg": [1.0] * 100})
    report = compare(ref, live).set_index("feature")
    assert report.loc["category", "status"] == "alert"   # 100% "other"
    assert report.loc["amount", "status"] == "alert"     # 100% missing
