"""Fast scoring for ONE payment at a time (the live API path).

pandas is built for millions of rows. For a single row, building a DataFrame
and passing it to the model costs ~3 ms of overhead, while the model itself
needs ~0.02 ms. So the live path builds a plain number array instead.

It must give EXACTLY the same numbers as the pandas code used in training;
tests/test_scoring.py checks this on thousands of real payments.
"""

import heapq

import lightgbm as lgb
import numpy as np

from sentinel.features import haversine_km
from sentinel.train import FEATURES


def top_reasons(contrib: np.ndarray, top_n: int = 3) -> list[tuple[str, float]]:
    """Clues that pushed the score UP the most (a heap picks the top n)."""
    top = heapq.nlargest(top_n, range(len(FEATURES)), key=lambda i: contrib[i])
    return [(FEATURES[i], float(contrib[i])) for i in top if contrib[i] > 0]


class Scorer:
    """Turns one payment + its card history into a score and reasons."""

    def __init__(self, booster: lgb.Booster, categories: list[str]) -> None:
        self.booster = booster
        # LightGBM stores categories as numbers: their position in the training list.
        self.category_code = {name: i for i, name in enumerate(categories)}

    def vector(self, payment: dict, history: dict) -> np.ndarray:
        """All clues, in the exact column order the model was trained with."""
        ts = payment["ts"]
        hour = ts.hour
        values = {
            **history,
            "amount": payment["amount"],
            "age": payment["age"],
            "city_pop": payment["city_pop"],
            "category": self.category_code[payment["category"]],
            "hour": hour,
            "day_of_week": ts.weekday(),
            "is_night": int(hour >= 22 or hour < 4),
            "distance_km": float(haversine_km(payment["cust_lat"], payment["cust_lon"],
                                              payment["merch_lat"], payment["merch_lon"])),
        }
        return np.array([[values[name] for name in FEATURES]], dtype=np.float64)

    def score(self, x: np.ndarray) -> float:
        # num_threads=1: for one row, starting extra threads costs more than it saves,
        # and many requests run at the same time anyway.
        return float(self.booster.predict(x, num_threads=1)[0])

    def reasons(self, x: np.ndarray, top_n: int = 3) -> list[str]:
        contrib = self.booster.predict(x, pred_contrib=True, num_threads=1)[0, :-1]
        return [name for name, _ in top_reasons(contrib, top_n)]
