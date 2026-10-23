"""Real-time fraud scoring API.

Run from the project root:
    uvicorn sentinel.api:app --reload
Then open http://127.0.0.1:8000/docs
"""

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from sentinel.decisions import APPROVE, BLOCK, REVIEW, Policy
from sentinel.explain import explain, load_model
from sentinel.features import CardState, add_simple_features
from sentinel.train import FEATURES

POLICY = Policy(review_line=0.02, block_line=0.69)
LABELS = {APPROVE: "approve", REVIEW: "review", BLOCK: "block"}


class Payment(BaseModel):
    """What the shop sends us. FastAPI checks these rules automatically."""

    card_id: int
    ts: datetime
    amount: float = Field(gt=0)
    category: str
    age: int = Field(ge=0, le=120)
    city_pop: int = Field(ge=0)
    cust_lat: float
    cust_lon: float
    merch_lat: float
    merch_lon: float


class Decision(BaseModel):
    """What we send back."""

    score: float
    decision: str
    reasons: list[str]
    latency_ms: float


state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts: load the model ONE time."""
    state["model"] = load_model()
    state["cards"] = defaultdict(CardState)  # card memory (moves to Redis later)
    yield


app = FastAPI(title="Sentinel fraud scoring", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Lets Kubernetes (later) check the API is alive."""
    return {"status": "ok"}


@app.post("/score", response_model=Decision)
def score(payment: Payment) -> Decision:
    start = time.perf_counter()
    card = state["cards"][payment.card_id]

    row = payment.model_dump()
    row.update(card.features(payment.ts, payment.amount))  # 1. clues from the past
    card.update(payment.ts, payment.amount)                # 2. remember this payment

    df = add_simple_features(pd.DataFrame([row]))
    df["category"] = pd.Categorical(df["category"])

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