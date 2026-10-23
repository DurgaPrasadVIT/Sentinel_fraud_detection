"""Data drift: is live traffic still like the data the model was trained on?

A model learns patterns from its training data. If live payments start to look
different (new shop categories, much bigger amounts, a new kind of attack), the
model's scores silently become less reliable. Drift checks catch that early.

Method: PSI (Population Stability Index), the standard in banking.
1. At training time, cut each clue into ~10 buckets using the training data
   (deciles), and remember what share of payments fell in each bucket.
2. Live, put recent payments into the same buckets.
3. PSI = sum over buckets of (live% - train%) * ln(live% / train%)

Rule of thumb used by credit-risk teams:
    PSI < 0.1  no real change | 0.1 - 0.25  moderate shift, watch it | > 0.25  significant shift
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PSI_WATCH = 0.1
PSI_ALERT = 0.25
MISSING = "missing"
EPS = 1e-4  # avoids log(0) when a bucket is empty


def numeric_edges(values: pd.Series, n_bins: int = 10) -> list[float]:
    """Inner bucket edges from training quantiles (duplicates removed)."""
    qs = np.nanquantile(values.astype(float), np.linspace(0, 1, n_bins + 1)[1:-1])
    return sorted(set(float(q) for q in qs))


def bucket_shares(values, edges: list[float] | None, categories: list[str] | None) -> dict[str, float]:
    """Share of values per bucket. Missing values (no history yet) get their own bucket."""
    s = pd.Series(values)
    total = max(len(s), 1)
    out = {}
    if categories is not None:
        counts = s.astype("object").where(s.notna(), MISSING).value_counts()
        for name in [*categories, MISSING, "other"]:
            out[name] = 0.0
        for name, count in counts.items():
            key = name if name in out else "other"
            out[key] += count / total
        return out
    x = pd.to_numeric(s, errors="coerce").astype(float)
    missing = x.isna()
    idx = np.searchsorted(edges, x[~missing].to_numpy(), side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    for i, c in enumerate(counts):
        out[f"b{i}"] = c / total
    out[MISSING] = missing.sum() / total
    return out


def psi(expected: dict[str, float], actual: dict[str, float]) -> float:
    total = 0.0
    for key in expected.keys() | actual.keys():
        e = max(expected.get(key, 0.0), EPS)
        a = max(actual.get(key, 0.0), EPS)
        total += (a - e) * math.log(a / e)
    return total


def build_reference(df: pd.DataFrame, features: list[str], scores: np.ndarray,
                    categorical: tuple[str, ...] = ("category",)) -> dict:
    """Snapshot of the training data's distributions, saved with the model."""
    ref = {"features": {}}
    for name in features:
        if name in categorical:
            cats = [str(c) for c in pd.Series(df[name]).astype("category").cat.categories]
            ref["features"][name] = {"categories": cats,
                                     "shares": bucket_shares(df[name].astype("object"), None, cats)}
        else:
            edges = numeric_edges(df[name])
            ref["features"][name] = {"edges": edges, "shares": bucket_shares(df[name], edges, None)}
    score_edges = numeric_edges(pd.Series(scores))
    ref["score"] = {"edges": score_edges, "shares": bucket_shares(scores, score_edges, None)}
    return ref


# Clues with a daily or weekly cycle can only be compared over a window that covers
# the whole cycle: 2 days of traffic contain only 2 weekdays, which is not drift.
MIN_SPAN_HOURS = {"day_of_week": 7 * 24, "hour": 24, "is_night": 24}


def compare(reference: dict, live: pd.DataFrame, live_scores=None,
            span_hours: float | None = None) -> pd.DataFrame:
    """PSI per clue (and for the model's scores), worst first.

    span_hours: how much time the live window covers. Cyclic clues are skipped when
    the window is shorter than their cycle.
    """
    rows = []
    for name, spec in reference["features"].items():
        if name not in live:
            continue
        if span_hours is not None and span_hours < MIN_SPAN_HOURS.get(name, 0):
            rows.append((name, np.nan))
            continue
        shares = bucket_shares(live[name], spec.get("edges"), spec.get("categories"))
        rows.append((name, psi(spec["shares"], shares)))
    if live_scores is not None:
        spec = reference["score"]
        rows.append(("model_score", psi(spec["shares"], bucket_shares(live_scores, spec["edges"], None))))
    out = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False, na_position="last")
    out["status"] = np.select([out["psi"].isna(), out["psi"] > PSI_ALERT, out["psi"] > PSI_WATCH],
                              ["skipped (window too short)", "alert", "watch"], "ok")
    return out.reset_index(drop=True)


def save_reference(ref: dict, path: Path) -> None:
    path.write_text(json.dumps(ref))


def load_reference(path: Path) -> dict:
    return json.loads(path.read_text())
