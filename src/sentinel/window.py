"""Sliding window over one card's recent payments.

Used to answer: "how many payments, and how much money,
did this card spend in the last N minutes?"
"""

from collections import deque
from datetime import datetime, timedelta


class SlidingWindow:
    """Keeps one card's payments from the last `window` of time."""

    def __init__(self, window: timedelta) -> None:
        self.window = window
        self.items: deque[tuple[datetime, float]] = deque()
        self.total = 0.0

    def _drop_old(self, now: datetime) -> None:
        """Remove payments that are `window` or older, from the front."""
        while self.items and self.items[0][0] <= now - self.window:
            _, old_amount = self.items.popleft()
            self.total -= old_amount

    def snapshot(self, now: datetime) -> tuple[int, float]:
        """Count and total of payments in the window BEFORE `now`."""
        self._drop_old(now)
        return len(self.items), self.total

    def add(self, ts: datetime, amount: float) -> None:
        """Record a new payment at the back of the line."""
        self.items.append((ts, amount))
        self.total += amount