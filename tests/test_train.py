"""Tests for sentinel.train."""

import pandas as pd

from sentinel.train import split_by_time


def test_split_by_time_has_no_overlap():
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2020-03-31 23:59", "2020-04-01 00:00", "2020-05-01 00:00"])}
    )
    train, valid = split_by_time(df, "2020-04-01")
    assert len(train) == 1
    assert len(valid) == 2
    assert train["ts"].max() < valid["ts"].min()  # train is fully before valid
