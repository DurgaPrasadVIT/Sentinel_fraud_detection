"""Real-time fraud scoring API.

Run from the project root:
    uvicorn sentinel.api:app --reload
Then open http://127.0.0.1:8000/docs

Card memory: set REDIS_URL (e.g. redis://localhost:6379/0) to keep it in Redis.
Without it, memory lives inside this process and is lost on restart.
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, get_args

import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field, field_validator

from sentinel.decisions import APPROVE, BLOCK, REVIEW, Policy
from sentinel.explain import explain, load_model
from sentinel.features import add_simple_features
from sentinel.store import make_store
from sentinel.train import FEATURES

POLICY = Policy(review_line=0.02, block_line=0.69)
LABELS = {APPROVE: "approve", REVIEW: "review", BLOCK: "block"}

# The 14 shop categories the model was trained on. Anything else is rejected
# with 422, instead of being silently scored as "unknown".
Category = Literal[
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]
CATEGORIES = list(get_args(Category))

EXAMPLE_PAYMENT = {
    "card_id": 4242,
    "ts": "2020-12-21T21:00:00",
    "amount": 25.0,
    "category": "gas_transport",
    "age": 35,
    "city_pop": 50000,
    "cust_lat": 40.72,
    "cust_lon": -74.04,
    "merch_lat": 40.75,
    "merch_lon": -74.0,
}


class Payment(BaseModel):
    """What the shop sends us. FastAPI checks these rules automatically."""

    model_config = {"json_schema_extra": {"examples": [EXAMPLE_PAYMENT]}}

    card_id: int
    ts: datetime
    amount: float = Field(gt=0)
    category: Category
    age: int = Field(ge=0, le=120)
    city_pop: int = Field(ge=0)
    cust_lat: float = Field(ge=-90, le=90)
    cust_lon: float = Field(ge=-180, le=180)
    merch_lat: float = Field(ge=-90, le=90)
    merch_lon: float = Field(ge=-180, le=180)

    @field_validator("ts")
    @classmethod
    def to_utc_without_timezone(cls, ts: datetime) -> datetime:
        """Store every time the same way: UTC, no timezone label.

        Training data has no timezones. Mixing "2020-12-21T21:00:00Z" and
        "2020-12-21T21:00:00" for the same card used to crash the API.
        """
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts


class Decision(BaseModel):
    """What we send back."""

    score: float
    decision: str
    reasons: list[str]
    latency_ms: float


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load the model and connect to card memory."""
    state["model"] = load_model()
    state["cards"] = make_store()
    yield


app = FastAPI(title="Sentinel fraud scoring", lifespan=lifespan)


@app.get("/health")
def health(response: Response) -> dict:
    """Lets Kubernetes (later) check the API and its card memory are alive."""
    store = state["cards"]
    try:
        store_ok = store.ping()
    except Exception:  # Redis down or unreachable
        store_ok = False
    if not store_ok:
        response.status_code = 503  # "not ready": send traffic elsewhere
    return {"status": "ok" if store_ok else "degraded", "store": type(store).__name__}


@app.post("/score", response_model=Decision)
def score(payment: Payment) -> Decision:
    start = time.perf_counter()
    row = payment.model_dump()
    # Clues from the card's past, then remember this payment (one atomic step in Redis).
    row.update(state["cards"].features_and_update(payment.card_id, payment.ts, payment.amount))

    df = add_simple_features(pd.DataFrame([row]))
    df["category"] = pd.Categorical(df["category"], categories=CATEGORIES)

    model = state["model"]
    fraud_score = float(model.predict(df[FEATURES])[0])
    decision = int(POLICY.decide(np.array([fraud_score]))[0])
    reasons = [] if decision == APPROVE else [name for name, _ in explain(model, df)[0]]

    return Decision(
        score=round(fraud_score, 4),
        decision=LABELS[decision],
        reasons=reasons,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )