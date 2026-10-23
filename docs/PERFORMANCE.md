# Performance: load test of the scoring API

## Setup
- Tool: [Locust](https://locust.io), `loadtest/locustfile.py`. Each simulated terminal sends a
  real payment from the test period and sends the next one as soon as the answer arrives.
- One API process (`uvicorn`, 1 worker), in-memory card store.
- Small machine: 2 vCPUs, **shared with the load generator itself**, so absolute numbers are
  pessimistic. The before/after comparison is what matters.

## Where the time went (one payment, before)
| Step | Time |
|---|---|
| Build a 1-row pandas DataFrame + simple clues | 1.98 ms |
| Model predict on the DataFrame | 1.24 ms |
| Model predict on a plain NumPy array | **0.02 ms** |
| SHAP reasons (only for review/block, ~1.5% of payments) | 3.3-4.5 ms |

pandas is built for large tables; for a single row its overhead was ~99% of the model time.

## The fix
`src/sentinel/scoring.py` builds the clues straight into a NumPy array in the model's column
order, and predicts with `num_threads=1` (starting threads for one row costs more than it saves).

Correctness is protected by tests: on 3,000 real payments the fast path gives
**identical clues, identical scores and identical SHAP reasons** to the pandas path used in
training (`tests/test_scoring.py`, tolerance 1e-12).

## Results
**Moderate load: 5 terminals**

| | Throughput | p50 | p99 |
|---|---|---|---|
| Before | 112 req/s | 43 ms | 83 ms |
| After | **1,002 req/s** | **4 ms** | **10 ms** |

**Heavy load: 50 terminals, no pause**

| | Throughput | p50 | p99 |
|---|---|---|---|
| Before | 107 req/s | 450 ms | 680 ms |
| After | **1,126 req/s** | **40 ms** | **110 ms** |

0 failed requests in every run.

**Developer laptop, real Redis in Docker** (Ryzen 5, 6 cores, 20 terminals, no pause)

| Throughput | p50 | p99 |
|---|---|---|
| **900 req/s** | **14 ms** | **30 ms** |

50,524 requests, 0 failures. Every payment includes a Redis round trip for card history.

Under heavy load, the remaining latency is **queueing**: 50 requests waiting for one Python
process on a shared 2-CPU machine. Each request itself takes ~4 ms.

## Scaling further
- **More worker processes:** `uvicorn sentinel.api:app --workers 4`. This needs `REDIS_URL`:
  with the in-memory store each process would keep its own, separate card history.
- **More API copies** behind a load balancer (Kubernetes, later), all sharing Redis.
- Redis adds a network round trip (~0.3-1 ms locally) per payment.

## Reproduce
```
uvicorn sentinel.api:app --log-level warning
locust -f loadtest/locustfile.py --headless -u 50 -r 25 -t 30s --host http://127.0.0.1:8000
```
Without `--headless`, open http://localhost:8089 for live charts.


## Investigation: 48 ms round trips in Docker

**Symptom.** With the full stack in Docker, the simulator saw ~48 ms per request, while the API
reported only ~4.6 ms p99 of its own processing time.

**Measure, don't guess.** `python -m sentinel.probe` sends sequential requests and splits the time:

| (Docker, API with `--workers 2`) | p50 |
|---|---|
| API's own processing time | 1.3 ms |
| Round trip, **reused** connection | **44.0 ms** |
| Round trip, **new** connection each time | 3.8 ms |

Reusing a connection was ~10x *slower* than opening a new one, the opposite of normal.

**First guess was wrong.** Nagle's algorithm on the client? The HTTP client already sets
`TCP_NODELAY` (checked its source). Ruled out.

**Reproduced outside Docker** by changing one thing, the worker count:

| uvicorn workers | Reused connection p50 |
|---|---|
| 1 | 1.4 ms |
| 2 | 43.0 ms |

**Root cause.** asyncio enables `TCP_NODELAY` on accepted connections only when the listening
socket has `proto == IPPROTO_TCP`. With one worker, asyncio creates the socket (`proto = 6`). With
several workers, uvicorn creates it itself with `proto = 0`, so `TCP_NODELAY` stays off. The server
then holds back the second part of each response (Nagle's algorithm) until the client acknowledges
the first, and the client delays that acknowledgement by ~40 ms ("delayed ACK"). New connections
escape it because TCP acknowledges immediately at the start of a connection.

**Fix.** One uvicorn process per container; scale with more containers (replicas). This is also the
usual container practice: one process, one health check, the orchestrator handles scaling.

**Lessons.** A *constant* extra delay (here ~40 ms whatever the load) points at protocol timing, not
slow code. Splitting server time from round-trip time found where to look; changing one variable at
a time found the cause.
