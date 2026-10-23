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

---

## Event streaming with Kafka (`events.py`, `simulator.py`, `monitor.py`)

### Where Kafka sits, and why there
```
card network (simulator) --HTTP--> scoring API --answer--> card network
                                        |
                                        +--copy of decision--> Kafka "decisions" topic
                                                                  |
                                       monitor / case notes / ring detector / training data
```
The shop is **waiting** for approve/block, so scoring is a direct HTTP call (like a
phone call). Everything that can happen later (dashboards, case notes, fraud-ring
detection, saving data for retraining) reads a **copy** of the decision from Kafka
(like a letter in a mailbox). The shop never waits for those slow jobs.

Putting Kafka *in front* of the API would force the shop to wait for an answer that
is sitting in a queue. That fits checks where nobody is waiting (e.g. AML pattern
analysis over hours), not card authorisation.

### Key design choices
- **Key = card id.** Kafka sends every event with the same key to the same partition,
  in order. So every consumer sees one card's story in the right sequence.
- **6 partitions.** Up to 6 copies of a consumer (same `group.id`) can share the work.
- **Non-blocking publish.** `produce()` only puts the event in a local queue; a
  background thread sends it. If Kafka is slow or down, the queue fills and we **drop
  the copy, never the payment** (`BufferError` is caught and counted).
- **`acks=all` + idempotence.** Kafka confirms the event is safely stored, and retries
  never create duplicates.
- **`flush()` on shutdown.** Events still in the queue get up to 5 seconds to be sent.
- **Consumer groups remember their position.** Stop the monitor and start it again:
  it continues where it stopped.

### The simulator (the "card network")
- Replays real payments from the test period, in time order, into the API.
- **Token bucket** limits the rate (DSA): a bucket of tokens refills at `rate` per
  second; each payment takes one; empty bucket = wait. Allows small bursts.
- **One card always goes through the same worker thread** (`card_id % workers`), so
  a card's payments arrive in order and its Redis history stays correct.
- **Warm-up** (`--warmup-days`): first replays a few days to build card history.
  Without it every card starts "brand new" and scores look worse than reality
  (the cold-start problem).
- **`--attack`** injects a stolen-card pattern every 500 payments: two big online
  purchases, then four tiny gas payments to test if the card still works.

### First end-to-end results (3,000 payments after 3 days warm-up, 1 API process)
- Fraud stopped: **36 of 42** payments ($14,424 of $17,875)
- Injected attack payments stopped: **25 of 30**. Every attack was stopped by its
  2nd payment at the latest; what got through was mostly the *first* $850 purchase on
  cards whose owners normally spend a lot. Fraud models catch **patterns**, and one
  purchase isn't a pattern yet.
- Throughput ~108 payments/s; client round-trip p50 70 ms, p99 130 ms. The API itself
  takes ~10 ms per payment; the rest is **queueing** in a single Python process with
  8 senders. Fixing that (more worker processes, faster model serving) is the
  load-testing step.
- Results are lower than the offline evaluation (98% of fraud money stopped) because
  only 3 days of history were replayed. Offline, every card had its full history.

### Run it yourself
```
docker compose up -d
$env:REDIS_URL = "redis://localhost:6379/0"
$env:KAFKA_BOOTSTRAP = "localhost:9092"
uvicorn sentinel.api:app
```
Second terminal: `python -m sentinel.monitor`
Third terminal: `python -m sentinel.simulator --rate 50 --limit 3000 --warmup-days 3 --attack`

### Likely interview questions
1. Why is Kafka after the API and not in front of it?
2. How do you keep one card's events in order? *(Partition by card id.)*
3. What happens to scoring if Kafka goes down? *(Nothing: publish is non-blocking;
   we drop the copy and count it.)*
4. At-least-once vs exactly-once: what do `acks=all` and idempotence give you?
5. How does a consumer resume after a crash? *(Committed offsets per consumer group.)*
6. Why did live results differ from offline results? *(Cold start: less card history.)*

---

## Load testing and the fast scoring path (`scoring.py`, `loadtest/`, `docs/PERFORMANCE.md`)

### What we did
1. **Measured first.** A load test with Locust: 50 simulated terminals sending payments
   non-stop. Result: 107 req/s, p99 680 ms. Far from the 50 ms target.
2. **Profiled** one payment step by step. 99% of the time was pandas overhead on a
   1-row table; the model itself needs 0.02 ms.
3. **Fixed** with a NumPy fast path, and **proved it gives identical results** on 3,000
   real payments.
4. **Measured again:** ~10x throughput (1,126 req/s) and p99 10 ms at moderate load.

### Lessons
- **Never optimise without measuring.** The guess "the model is slow" was wrong: the model
  was the fastest part.
- **Tools have a sweet spot.** pandas is fast for a million rows and slow for one.
- **A speed-up is only valid if results stay identical.** The parity test is what makes this
  change safe.
- **Latency under load = work + waiting in line.** Once the work is fast, more processes or
  machines reduce the waiting. Multiple processes need shared state (Redis).

### Likely interview questions
1. How did you find the bottleneck? *(Load test, then timing each step.)*
2. Why was pandas slow here? *(Per-call overhead dominates for 1 row.)*
3. How do you know the optimisation didn't change model behaviour? *(Parity test.)*
4. What's the difference between throughput and latency? Why do both matter?
5. How would you scale to 10,000 payments/s? *(More processes and API copies, shared Redis,
   partitioned Kafka, maybe a compiled model runtime such as ONNX or Treelite.)*

---

## Personal clues and the travel-speed clue (`features.py`: `speed_kmh`)

### What we tried, and what the data said
Idea: "is this payment normal **for this card**?" Candidate clues, checked on all 1.85M payments
(median, normal vs fraud) before writing any production code:

| Candidate | Normal | Fraud | Verdict |
|---|---|---|---|
| amount vs card's mean, in std units | -0.19 | 2.47 | strong alone, but *hurt* the model (overlaps `amount_vs_avg`) |
| amount vs card's max so far | 0.02 | 0.30 | same |
| share of card's payments at this hour | 0.047 | 0.042 | no signal |
| share of card's payments in this category | 0.081 | 0.080 | no signal |
| km from the card's previous shop | 101 | 100 | no signal |
| **travel speed from the previous shop (km/h)** | **21** | **64** | **kept** |

The synthetic data gives cards no personal habits and places shops randomly, so hour/category
habits and true "impossible travel" can't be demonstrated with it. The code still computes
real travel speed: on real data, New York at 10:00 then London at 10:15 gives >10,000 km/h.

### Don't trust one training run
Adding `speed_kmh`, compared over **4 random seeds**:
- PR-AUC higher in 4 of 4 seeds (+0.0013 to +0.0047)
- money lost lower in 4 of 4 seeds, **~12% less on average**
- good customers blocked: **mixed** (better in 2 seeds, worse in 2), so we don't claim it

The released model (seed 42) shows $25,268 -> $18,342 (-27%), but seed 42 happened to be the
weakest baseline. The honest number to quote is the **~12% average**.

### Bug this exposed: hard-coded decision lines
The block line (0.69) was typed into the API. After retraining, the best line became 0.85.
Without a fix, the API would have silently used lines tuned for a different model.
Now `python -m sentinel.decisions` saves `models/policy.json` with a **fingerprint of the model
file**, and the API refuses to start if the policy belongs to another model.

### Checks
- `speed_kmh` from the training loop matches an independent vectorised pandas version on all
  1,852,394 payments: 0 mismatches.
- Redis store vs training logic: parity test now includes shop locations.
- Live simulator run (same settings as before): fraud stopped 36/42 -> 38/42, injected attack
  payments 25/30 -> 27/30.

### After every retrain
```
python -m sentinel.features   # rebuild clues
python -m sentinel.train      # train and save the model
python -m sentinel.decisions  # re-choose approve/review/block lines, saved with the model
```

### Likely interview questions
1. A new feature improved your metric. How do you know it's real? *(Several seeds; report the
   average; only claim what is consistent.)*
2. Why can adding a strong feature make a model worse? *(Redundant with existing features,
   adds noise; early stopping hit sooner.)*
3. What is "impossible travel" and what data do you need for it?
4. How do you stop a model and its thresholds from getting out of sync? *(Version them together;
   fingerprint; fail at startup.)*

---

## Fraud rings and money mules (`rings.py`, device rule in `api.py`)

### Why a graph
A single-payment model judges each payment alone. A **mule ring** (several stolen cards
cashed out from a few shared phones, with small "low and slow" payments) looks normal one
payment at a time. The connections give it away:
```
card A -- device 1 -- card B -- device 2 -- card C
```
A and C never shared a device, but they're in the same group. Finding groups =
**connected components** of the card-device graph.

### Union-find (DSA)
- `find(x)`: which group is x in. `union(a, b)`: merge two groups.
- **Path compression** + **union by size** make each operation almost O(1), so it works on
  millions of cards and devices, and it updates **incrementally** as payments stream in.
- Each group keeps its card list; when groups merge, the smaller list moves into the bigger
  one ("small-to-large"), so each card moves at most log2(n) times.
- BFS/DFS would also find components, but must re-scan the graph; union-find just updates.

### Two layers
1. **Real-time rule in the API** (Redis sorted set per device): a device used by 5+ different
   cards in 30 days sends the payment to review, with the reason `device_used_by_N_cards`.
2. **Graph detector** (`python -m sentinel.rings`) reads the Kafka decisions stream, alerts
   when a group reaches 5 cards, and publishes the alert to the `ring-alerts` topic for a
   case-management system.

### Honest test setup (red-team injection)
The public dataset has **no devices and no rings** (checked: no common point of compromise,
no coordinated timing beyond normal night-time fraud). So the simulator:
- gives every card 1-2 own devices, and puts ~3% of cards in **households of 2-3 that share
  a family tablet** (normal behaviour the detector must not flag);
- injects mule rings: 6-9 real cards (keeping their owners' details), 2 shared devices,
  3 payments each of $40-250 over 12 hours.

### Results (3,000 real payments + 5 rings + card-testing attacks)
- **Graph detector: 5/5 rings found, 35/35 ring cards, 0 innocent cards flagged.**
- **At payment time only 8-10 of 105 ring payments were stopped.** Rings spread their cards
  over 2 devices, so each device stays under the 5-card rule, and the model sees small normal
  amounts. This is the point: the graph sees what per-payment checks can't.
- **Timeliness: 17% of ring money came after the alert** (stoppable by freezing the ring's
  cards). The alert waits for the 5th card; by then most payments are done.

### Known limits / next steps
- Synthetic benign behaviour is simple (households <= 3 cards), so 0 false flags is easy here.
  Real data has public Wi-Fi, shared work devices and device-ID collisions: you'd add device
  reputation, edge weights, time decay (forget old links) and a review queue for alerts.
- Earlier alerts: score groups by risk (fraction of cards with risky payments, new devices)
  instead of a fixed card count.
- The graph lives in one process's memory; at scale it would be sharded or kept in a graph
  store, rebuilt from Kafka after a restart.

### Likely interview questions
1. Why would a per-payment model miss a mule ring?
2. Union-find vs BFS for connected components on a stream?
3. How do you avoid flagging families that share a tablet?
4. Your detector found every ring but only 17% of the money was after the alert. How
   would you alert earlier, and what does it cost?
5. Your data had no rings. How did you test a ring detector honestly?

---

## AI case notes (`casenotes.py`, `cases.py`)

### What it does
For every payment sent to review or blocked, analysts get a short note: what happened, why
it looks risky, what to do next. Example (template writer, real output):
> BLOCK (risk 100.0%): $326.94 grocery_pos on card ****5478 at 2020-06-22 01:25. Why: amount
> $326.94; $4,822.58 spent on this card in the previous 24 hours; category grocery_pos.
> Next step: Contact the cardholder; if they don't recognise it, close and reissue the card.

### Design decisions
- **Not on the payment path.** The case service reads decisions from Kafka. A slow or broken
  LLM can't slow down or break payments. Notes are written on 4 background threads.
- **Facts first, LLM second.** We turn the model's SHAP reasons + clue values into plain-English
  facts ourselves. The LLM only rephrases them.
- **Hallucination guard.** Every number in an LLM note must match a number in the facts
  (rounding allowed). Invented numbers, long digit strings (possible card numbers), errors or
  timeouts -> the template note is used, and the rejection reason is saved for review.
- **Evaluated with real LLM text** (`tests/test_note_guard_eval.py`): 8 notes written by an
  LLM (Claude) from real cases, with natural rounding ("$2,081", "about 5 times"), must be
  accepted (8/8); 10 realistic hallucinations must be rejected (10/10). The first version of
  the guard **missed 4 of them**: a wrong time ("02:32" for 22:32, checked digit by digit), an
  invented "2 previous chargebacks" (small numbers were always allowed), "in Paris", and
  "already reported stolen". Fixed by checking times/dates as a whole, removing the
  small-number loophole, rejecting unknown capitalised names, and rejecting claims about data
  we don't have (reports, chargebacks, police...).
- **Limit:** a word-level check can't catch every invented claim. Stronger options: structured
  output (the LLM picks from given facts, code writes the numbers), or a second model that
  verifies each sentence against the facts.
- **Real API check:** the Anthropic writer was called against the real endpoint without a key:
  401 Unauthorized -> template note, reason saved. This proves connectivity and the fallback,
  not the request format (the server checks the key first).
- **Privacy.** The LLM only sees `****1234`. With Ollama, nothing leaves the machine at all.
- **Fairness.** The model uses age and town size. Notes never cite them: "risky because the
  cardholder is 61" invites biased decisions and describes the person, not the behaviour.
  The model's use of age is a known issue for a fairness review (e.g. compare false-block
  rates across age groups, try the model without it).
- **Pluggable writers:** `SENTINEL_LLM=template` (default) | `ollama` | `anthropic` | `openai`.
- **At-least-once + idempotent saves.** Kafka may re-deliver after a crash; `txn_id` primary
  keys make duplicates harmless. The Kafka position is committed only after rows are saved.
- **SQLite with WAL** so the dashboard can read while the service writes. PostgreSQL in
  production; the code would barely change.

### Try a writer on one sample case
```
python -m sentinel.casenotes                  # shows the facts, the note, and why an LLM note was rejected
```

### Run it
```
python -m sentinel.cases                      # template notes
$env:SENTINEL_LLM = "anthropic"; $env:ANTHROPIC_API_KEY = "<your key>"; python -m sentinel.cases   # Claude
$env:SENTINEL_LLM = "ollama"; python -m sentinel.cases   # local LLM (install Ollama, `ollama pull llama3.1:8b`)
```

### Likely interview questions
1. How do you stop an LLM from inventing facts in a regulated setting?
2. Why is the LLM not called inside the scoring API?
3. What happens if the LLM is down?
4. Exactly-once vs at-least-once: how do you avoid duplicate cases?
5. Your model uses age. What would you do about it?

---

## Live operations dashboard (`dashboard.py`)

### What's on it
- **Live numbers** (refresh every 3 s): payments scored, block / review share, money held for
  checks, API p99 latency, open cases and ring alerts.
- **Decisions per minute** chart.
- **Ring alerts** from the graph detector (cards masked).
- **Review queue**, highest risk first: each case shows its note, the model's reasons, and who
  wrote the note (template or which LLM).
- **Analyst buttons: "Confirm fraud" / "Not fraud".** These labels are the most valuable data
  a fraud team has: they are how the model gets retrained on new fraud patterns
  (human-in-the-loop). They also measure false alarms.

### Design
- The dashboard only **reads the case database**; it never talks to Kafka or the API, so a
  busy dashboard can't affect payments.
- Data functions (`summary`, `review_queue`, ...) are plain Python and unit-tested; the page
  itself has a smoke test with Streamlit's `AppTest` (it caught a crash on a fresh machine
  with no `data/` folder).
- `st.fragment(run_every="3s")` refreshes only the live part of the page, not the whole app.
- Streamlit was chosen because it's pure Python and fast to build. A bank would build the
  analyst tool in a web framework (React etc.) on top of the same database/API.

### Run the whole system
```
docker compose up -d
uvicorn sentinel.api:app --log-level warning      # with REDIS_URL and KAFKA_BOOTSTRAP set
python -m sentinel.rings
python -m sentinel.cases
streamlit run src/sentinel/dashboard.py
python -m sentinel.simulator --rate 50 --limit 5000 --warmup-days 3 --attack --rings 5
```

### Likely interview questions
1. Why do analyst labels matter more than the model's own predictions for retraining?
2. Why shouldn't the dashboard read from the scoring API directly?
3. How would you measure analyst workload and false alarms from this data?

---

## CI and Docker (`.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml`)

### CI: every PR is checked automatically
GitHub Actions runs on every pull request and push to `main`:
1. install exact library versions (`requirements-dev.txt`)
2. **lint** with ruff (unused imports, bugs like `zip()` silently dropping items, style)
3. **tests**
4. **build the Docker image** and import every service inside it

**The problem we hit:** the dataset and trained model aren't in git, so in a clean CI machine
**all 17 API tests were skipped**: CI would never test the API. Fix (`tests/conftest.py`): when
no model exists, train a tiny stand-in model on random data (~1 s) before tests are collected.
Result in a clean environment: **100 passed, 2 skipped** (the 2 need the real 1.85M-row data).

### Pinned versions
`requirements.txt` used to say just `pandas`. A new release could change behaviour silently, and
CI, Docker and your laptop could each run different versions. Now every library has an exact
version. Runtime libraries are in `requirements.txt` (what the Docker image needs); test, lint
and notebook tools in `requirements-dev.txt`.

### One image, many services
The same image runs the API, case service, ring detector, dashboard and simulator; compose
picks the command. Details:
- **The model is not in the image.** `models/` is mounted at run time: a new model is deployed
  without rebuilding code. (In a bigger setup, the API would pull it from a model registry.)
- **Not root.** Containers run as user `sentinel`.
- **Health check** on the API, so Docker (and later Kubernetes) knows when it's ready.
- **Kafka has two listeners:** `localhost:9092` for your laptop, `kafka:19092` for containers.
  With one, containers would be told to connect to "localhost" (themselves) and fail.
- **The case database lives on a Docker volume**, shared by the case service and the dashboard.
  SQLite's WAL mode shares memory between processes, which is unreliable across containers on
  Windows folder mounts.

### Bug found on the first real run: volume permissions
The dashboard showed `sqlite3.OperationalError: unable to open database file`. Containers run as
the non-root user `sentinel`, but the folder the `cases` volume is mounted on didn't exist in the
image, so Docker created it **owned by root**, and `sentinel` couldn't write there. Fix: create
`/app/cases` in the Dockerfile, owned by `sentinel` (a new volume copies that ownership). CI now
runs the image like compose does (non-root, fresh volume) and opens the database, so this can't
come back unnoticed. Lesson: building an image isn't testing it; run it the way production does.

### Second bug on the first real run: services starting in a different order
The dashboard opened but showed 0 payments. In docker compose all services start together; the
case service subscribed to `decisions` **before the API had created the topic**. A Kafka consumer
only re-checks for new topics every few minutes, and with `auto.offset.reset=latest` it skipped
everything sent meanwhile. Fix: every consumer creates the topics it reads before subscribing.
Lesson: distributed services must not depend on start-up order.

### Run everything with Docker
```
docker compose --profile app up -d --build
docker compose --profile demo run --rm simulator
```
API: http://localhost:8000/docs, dashboard: http://localhost:8501

### Likely interview questions
1. Why not put the model file in the Docker image?
2. Your tests need a 300 MB dataset and a trained model. How do you test in CI?
3. Why pin exact versions? What's the downside? *(Must update deliberately; use a bot like
   Dependabot.)*
4. Why does Kafka need two listeners in Docker?
