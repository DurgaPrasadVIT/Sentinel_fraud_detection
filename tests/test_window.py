"""Tests for sentinel.window."""

from datetime import datetime, timedelta

from sentinel.window import SlidingWindow

T0 = datetime(2020, 1, 1, 12, 0)
ONE_HOUR = timedelta(hours=1)


def test_empty_window():
    w = SlidingWindow(ONE_HOUR)
    assert w.snapshot(T0) == (0, 0.0)


def test_counts_recent_payments():
    w = SlidingWindow(ONE_HOUR)
    w.add(T0, 10.0)
    w.add(T0 + timedelta(minutes=10), 20.0)
    assert w.snapshot(T0 + timedelta(minutes=30)) == (2, 30.0)


def test_drops_old_payments():
    w = SlidingWindow(ONE_HOUR)
    w.add(T0, 10.0)
    w.add(T0 + timedelta(minutes=50), 20.0)
    # 70 minutes later, the first payment is more than 1 hour old.
    assert w.snapshot(T0 + timedelta(minutes=70)) == (1, 20.0)


def test_exactly_one_hour_old_is_dropped():
    w = SlidingWindow(ONE_HOUR)
    w.add(T0, 10.0)
    assert w.snapshot(T0 + ONE_HOUR) == (0, 0.0)


def test_snapshot_does_not_include_current_payment():
    w = SlidingWindow(ONE_HOUR)
    w.add(T0, 10.0)
    now = T0 + timedelta(minutes=5)
    count, total = w.snapshot(now)   # clues first...
    w.add(now, 99.0)                 # ...then add the new payment
    assert (count, total) == (1, 10.0)