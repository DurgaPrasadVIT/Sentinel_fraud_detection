"""Load test for the scoring API.

Simulates many card terminals sending payments at the same time.
Run headless from the project root (API must be running):
    locust -f loadtest/locustfile.py --headless -u 50 -r 10 -t 60s --host http://127.0.0.1:8000

-u = number of simulated terminals, -r = how many to start per second, -t = duration.
Or without --headless, then open http://localhost:8089 for live charts.
"""

import random

import pandas as pd
from locust import HttpUser, constant, task

from sentinel.data import PROCESSED_DIR
from sentinel.simulator import to_payment

# Real payments, so the API does real work (history, model, and SHAP for risky ones).
_PAYMENTS = [
    to_payment(row)
    for row in pd.read_parquet(PROCESSED_DIR / "fraudTest.parquet").sample(20_000, random_state=0)
    .to_dict("records")
]


class Terminal(HttpUser):
    wait_time = constant(0)  # send the next payment as soon as the answer arrives

    @task
    def score_payment(self):
        payment = random.choice(_PAYMENTS)
        with self.client.post("/score", json=payment, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
