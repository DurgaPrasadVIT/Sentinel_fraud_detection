# Learning notes

Plain-English notes for each part of Sentinel, with likely interview questions.
Read one section at a time, next to the code it describes.

---

## Card memory in Redis (`src/sentinel/store.py`)

### What problem it solves
The API needs each card's recent history (payments in the last hour, day, week)
to build clues. At first this lived in a Python dict inside the API. Two problems:

1. **Restart the API and all history is gone.** A thief could wait for a restart
   and look like a brand-new, "clean" card.
2. **Real systems run many copies of the API.** Each copy would have its own dict,
   so one card's history would be split across copies.

Redis is one shared, in-memory database that every API copy talks to. It survives
API restarts and answers in well under a millisecond.

### How the data is stored
For each card, two Redis keys:

| Key | Redis type | What's inside |
|---|---|---|
| `card:<id>:recent` | sorted set | payments from the last 7 days. score = time, member = `seq:amount` |
| `card:<id>:stats` | hash | lifetime count `n`, `total` spent, `last_ts`, and a `seq` counter |

A **sorted set** keeps members ordered by score (here: time). That makes
"remove everything older than 7 days" a single command (`ZREMRANGEBYSCORE`).
It is the Redis version of the `deque` in our `SlidingWindow`.

### Why a Lua script?
Reading history and adding the new payment must happen as **one step**.
If two payments for the same card arrive at the same moment, without this they
could both read the history before either one adds itself, and both would miss
each other. A Lua script runs inside Redis **atomically**: nothing else can run
in the middle.

### Why `seq` in the member?
A sorted set can't hold the same member twice. Two $20 payments at the same
second would be the same `time:20.0` and the second would vanish. Adding a
counter (`seq`) makes every member unique.

### Why keys expire (TTL ~13 months)
Without expiry, Redis memory grows forever with cards that are never used again.
Every write refreshes the timer, so active cards never expire.

### How we know it's correct
`tests/test_store.py::test_redis_matches_training_features` feeds 400 random
payments to both the training logic (`CardState`) and Redis and checks every clue
matches. We also checked **50,000 real payments: 0 mismatches**.
This guards against **training-serving skew**: when live clues are calculated
differently from training clues, the model quietly gets worse and nobody notices.

### Same interface, two stores
`InMemoryCardStore` and `RedisCardStore` both have `features_and_update(...)`.
The API doesn't care which one it gets (`make_store()` picks Redis when
`REDIS_URL` is set). Tests use the in-memory one or `fakeredis`, so they need no
real Redis server.

### Known limits (say these in interviews, they show maturity)
- **Out-of-order payments:** the script trims history using the newest payment's
  time. If a payment arrives late (older timestamp), its window may be slightly off.
  Real systems handle this with event-time processing (e.g. Kafka + Flink).
- **Blocked attempts are remembered too.** That's deliberate (repeated attempts
  are a fraud signal), but some banks track "attempts" and "approved spend" separately.
- **Redis down = API not ready.** `/health` returns 503 so a load balancer or
  Kubernetes stops sending traffic. A further step would be a fallback (e.g. score
  with only the payment's own clues and send more cases to review).

### Likely interview questions
1. Why Redis and not PostgreSQL for features at scoring time?
   *Latency: Redis keeps data in RAM (sub-millisecond); a disk database is ~5-20 ms.*
2. How do you avoid a race condition when two payments hit the same card at once?
   *An atomic Lua script: read + write happen as one step.*
3. How do you know the live features match the training features?
   *A parity test on random data plus a check on 50,000 real payments: 0 mismatches.*
4. What happens if Redis goes down?
   *Health check fails (503), traffic is routed away; possible degraded mode.*
5. How do you stop Redis from running out of memory?
   *Trim to the longest window (7 days) on every write, plus a TTL on every key.*

### Try it yourself
```
docker start sentinel-redis
$env:REDIS_URL = "redis://localhost:6379/0"
uvicorn sentinel.api:app --reload
```
Send a few payments in `/docs`, stop the API (Ctrl + C), start it again, and send
another payment for the same card: the history is still there. Peek inside Redis:
```
docker exec -it sentinel-redis redis-cli HGETALL card:4242:stats
docker exec -it sentinel-redis redis-cli ZRANGE card:4242:recent 0 -1 WITHSCORES
```

---

## Input validation (`Payment` in `src/sentinel/api.py`)

### The bugs a manual test found
Sending our own made-up payment to `/docs` exposed two problems:

1. **Unknown categories were silently accepted.** `"electronic"` is not one of the
   14 categories the model was trained on. The model treated it as "missing" and
   still returned a score: a confident-looking answer built on a guess.
2. **Mixing timezones crashed the API.** `"...T21:00:00Z"` (with timezone) followed by
   `"...T21:05:00"` (without) for the same card raised a `TypeError` (HTTP 500),
   because Python can't compare the two kinds of datetime.

### The fixes
- `category` is now a `Literal` of the 14 known values, so anything else returns **422**
  with a clear message. A test checks this list matches the categories saved in the model
  file, so a retrain with new categories can't silently break it.
- Every timestamp is converted to **UTC with no timezone label**, the same format as the
  training data. A New York 21:00 payment becomes 02:00 UTC the next day.
- Latitude and longitude must be on Earth (-90..90, -180..180).
- The `/docs` page now shows a realistic example instead of zeros.

### Lesson
**Fail loudly on bad input.** A 422 error is annoying; a wrong fraud score that nobody
notices is dangerous. Validate at the edge of the system, before data reaches the model.

### About latency
The very first request after starting the server is slower (~60-100 ms): libraries warm
up on first use. After that: **p50 ≈ 10 ms, p99 ≈ 19 ms** (400 requests, in-memory store).
"p99" = 99% of requests are faster than this. Seniors quote p99, not the average, because
the slowest requests are the ones customers complain about.

### Likely interview questions
1. What happens if a client sends a category your model has never seen?
2. How do you handle timezones in a payments system? *(Store UTC everywhere.)*
3. Why do you report p99 latency instead of the average?
