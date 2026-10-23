"""Latency probe: where does the time go between a client and the API?

Sends sequential requests (no concurrency, no rate limit) and compares:
- server time: what the API measured itself (latency_ms in the response)
- round trip with a reused connection (keep-alive)
- round trip opening a new connection every time (includes DNS + TCP connect)

    python -m sentinel.probe --url http://127.0.0.1:8000/score
    docker compose --profile demo run --rm simulator python -m sentinel.probe --url http://api:8000/score
"""

import argparse
import statistics
import time

from sentinel.api import EXAMPLE_PAYMENT


def measure(client, url: str, n: int) -> tuple[list[float], list[float]]:
    round_trip, server = [], []
    for i in range(n):
        payment = {**EXAMPLE_PAYMENT, "card_id": 900_000 + i}  # fresh cards: no state build-up
        start = time.perf_counter()
        body = client.post(url, json=payment).json()
        round_trip.append((time.perf_counter() - start) * 1000)
        server.append(body["latency_ms"])
    return round_trip, server


def p(values: list[float]) -> str:
    q = statistics.quantiles(values, n=100)
    return f"p50 {statistics.median(values):6.1f} ms | p99 {q[98]:6.1f} ms"


def main() -> None:
    import httpx2 as httpx

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000/score")
    parser.add_argument("-n", type=int, default=100)
    args = parser.parse_args()

    with httpx.Client() as client:
        measure(client, args.url, 5)  # warm-up
        reused, server = measure(client, args.url, args.n)
    fresh = []
    for _ in range(args.n // 2):
        with httpx.Client() as client:
            fresh += measure(client, args.url, 1)[0]

    print(f"API's own processing time:          {p(server)}")
    print(f"Round trip, reused connection:      {p(reused)}")
    print(f"Round trip, new connection each:    {p(fresh)}")
    print(f"=> time outside the API (reused):   {statistics.median(reused) - statistics.median(server):6.1f} ms")


if __name__ == "__main__":
    main()
