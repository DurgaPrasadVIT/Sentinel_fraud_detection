"""Real-time fraud scoring API.

Run from the project root:
    uvicorn sentinel.api:app --reload
Then open http://127.0.0.1:8000/docs

Card memory: set REDIS_URL (e.g. redis://localhost:6379/0) to keep it in Redis.
Without it, memory lives inside this process and is lost on restart.

Decision events: set KAFKA_BOOTSTRAP (e.g. localhost:9092) to publish every
decision to the "decisions" topic. Without it, events are not published.
"""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, get_args

import numpy as np
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field, field_validator

from sentinel.decisions import APPROVE, BLOCK, REVIEW, Policy
from sentinel.events import make_publisher, utc_now
from sentinel.explain import load_model
from sentinel.scoring import Scorer
from sentinel.store import make_store

LABELS = {APPROVE: "approve", REVIEW: "review", BLOCK: "block"}

# Rule next to the model: a device used by this many different cards in 30 days
# is a typical sign of a fraud ring or money mules. Normal people use 1-2 devices,
# and a household may share one between 2-3 cards.
DEVICE_CARD_LIMIT = 5

# The 14 shop categories the model was trained on. Anything else is rejected
# with 422, instead of being silently scored as "unknown".
Category = Literal[
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]
CATEGORIES = list(get_args(Category))

EXAMPLE_PAYMENT = {
    "device_id": "a1b2c3d4e5f6",
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

    txn_id: str | None = Field(default=None, max_length=64)  # created by us if missing
    device_id: str | None = Field(default=None, max_length=128)  # phone / browser fingerprint
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

    txn_id: str
    score: float
    decision: str
    reasons: list[str]
    latency_ms: float


state = {}


def json_safe(values: dict) -> dict:
    """NaN is not valid JSON: send "no value yet" as null. Round long decimals."""
    out = {}
    for name, value in values.items():
        if isinstance(value, float):
            value = None if value != value else round(value, 4)
        out[name] = value
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load the model and connect to card memory."""
    state["scorer"] = Scorer(load_model(), CATEGORIES)
    state["policy"] = Policy.load()  # approve/review/block lines chosen for THIS model
    state["cards"] = make_store()
    state["events"] = make_publisher()
    yield
    state["events"].close()  # send any events still waiting in the queue


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
    history = state["cards"].features_and_update(
        payment.card_id, payment.ts, payment.amount, payment.merch_lat, payment.merch_lon
    )

    scorer = state["scorer"]
    clues = scorer.clues(row, history)
    x = scorer.to_array(clues)
    fraud_score = scorer.score(x)
    decision = int(state["policy"].decide(np.array([fraud_score]))[0])
    reasons = [] if decision == APPROVE else scorer.reasons(x)

    if payment.device_id:
        n_cards = state["cards"].cards_on_device(payment.device_id, payment.card_id, payment.ts)
        if n_cards >= DEVICE_CARD_LIMIT:
            decision = max(decision, REVIEW)  # a person checks it; never auto-approve
            reasons = [f"device_used_by_{n_cards}_cards", *reasons]

    result = Decision(
        txn_id=payment.txn_id or uuid.uuid4().hex,
        score=round(fraud_score, 4),
        decision=LABELS[decision],
        reasons=reasons,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    # Copy of the decision for everything downstream. Non-blocking.
    state["events"].publish({
        **payment.model_dump(exclude={"txn_id"}),
        **result.model_dump(),
        "clues": json_safe(clues),  # what the model saw: for case notes and drift checks
        "scored_at": utc_now(),
    })
    return result