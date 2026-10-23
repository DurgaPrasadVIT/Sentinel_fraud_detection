"""Tests for sentinel.features."""

import numpy as np
import pandas as pd
import pytest

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

def test_travel_speed_between_shops():
    df = make_df()
    df.loc[1, ["merch_lat", "merch_lon"]] = [51.5, -0.13]  # 2nd payment of card 1: London
    out = build_features(df)
    assert np.isnan(out.iloc[0]["speed_kmh"])         # first payment: nothing to compare
    assert out.iloc[1]["speed_kmh"] > 10_000          # New York -> London in 30 min: impossible
    assert out.iloc[3]["speed_kmh"] < 1_000           # back home 13.5 hours later: possible


def test_travel_speed_counts_very_short_gaps_as_one_minute():
    from sentinel.features import travel_speed_kmh

    one_km_east = -74.0 + 1 / 84.4  # ~1 km at this latitude
    assert travel_speed_kmh(40.0, -74.0, 40.0, one_km_east, secs=0) == pytest.approx(60, rel=0.05)
