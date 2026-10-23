"""Tests for sentinel.data."""

import pandas as pd

from sentinel.data import PII_COLUMNS, clean


def make_raw() -> pd.DataFrame:
    """A tiny fake raw dataset with the same columns as the real CSV."""
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.to_datetime(["2020-06-01 10:00", "2020-01-01 09:00"]),
            "cc_num": [111, 222],
            "merchant": ["shop_a", "shop_b"],
            "category": ["grocery_pos", "shopping_net"],
            "amt": [25.5, 900.0],
            "first": ["Asha", "Ravi"],
            "last": ["K", "M"],
            "gender": ["F", "M"],
            "street": ["1 Main St", "2 Oak Ave"],
            "city": ["Jersey City", "Newark"],
            "state": ["NJ", "NJ"],
            "zip": [7302, 7102],
            "lat": [40.72, 40.73],
            "long": [-74.04, -74.17],
            "city_pop": [260000, 310000],
            "job": ["Nurse", "Teacher"],
            "dob": pd.to_datetime(["1990-06-01", "2000-01-02"]),
            "trans_num": ["t1", "t2"],
            "unix_time": [1, 2],
            "merch_lat": [40.7, 40.8],
            "merch_long": [-74.0, -74.1],
            "is_fraud": [0, 1],
        }
    )


def test_removes_pii():
    out = clean(make_raw())
    for col in PII_COLUMNS:
        assert col not in out.columns


def test_renames_columns():
    out = clean(make_raw())
    for col in ["ts", "card_id", "amount", "txn_id"]:
        assert col in out.columns


def test_sorted_by_time():
    out = clean(make_raw())
    assert out["ts"].is_monotonic_increasing


def test_age_is_correct():
    out = clean(make_raw())
    # After sorting, row 0 is the 2020-01-01 payment. Born 2000-01-02,
    # so this person turns 20 tomorrow: age must be 19.
    assert out.loc[0, "age"] == 19
    # Row 1: born 1990-06-01, paying on 2020-06-01, exactly 30 today.
    assert out.loc[1, "age"] == 30


def test_does_not_change_input():
    raw = make_raw()
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)
