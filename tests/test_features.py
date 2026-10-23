"""Tests for sentinel.features."""

import numpy as np
import pandas as pd

from sentinel.features import build_features


def make_df() -> pd.DataFrame:
    """Card 1 pays 3 times, card 2 pays once, all at the same place."""
    return pd.DataFrame(
        {
            "card_id": [1, 1, 2, 1],
            "ts": pd.to_datetime(
                ["2020-01-01 22:00", "2020-01-01 22:30",
                 "2020-01-01 22:40", "2020-01-02 12:00"]
            ),
            "amount": [10.0, 30.0, 500.0, 20.0],
            "cust_lat": [40.0] * 4,
            "cust_lon": [-74.0] * 4,
            "merch_lat": [40.0] * 4,
            "merch_lon": [-74.0] * 4,
        }
    )


def test_first_payment_has_no_history():
    first = build_features(make_df()).iloc[0]
    assert first["count_1h"] == 0
    assert np.isnan(first["amount_vs_avg"])
    assert np.isnan(first["secs_since_last"])


def test_uses_only_earlier_payments_of_same_card():
    second = build_features(make_df()).iloc[1]
    assert second["count_1h"] == 1
    assert second["sum_1h"] == 10.0
    assert second["amount_vs_avg"] == 3.0     # 30 vs average 10
    assert second["secs_since_last"] == 1800  # 30 minutes


def test_cards_do_not_mix():
    card2 = build_features(make_df()).iloc[2]
    assert card2["count_24h"] == 0


def test_longer_windows():
    last = build_features(make_df()).iloc[3]
    assert last["count_1h"] == 0     # previous payments were 13+ hours ago
    assert last["count_24h"] == 2
    assert last["sum_24h"] == 40.0
    assert last["amount_vs_avg"] == 1.0  # 20 vs average 20


def test_night_and_distance():
    out = build_features(make_df())
    assert out.iloc[0]["is_night"] == 1   # 10 PM
    assert out.iloc[3]["is_night"] == 0   # 12 PM
    assert out.iloc[0]["distance_km"] == 0