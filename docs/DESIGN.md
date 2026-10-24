# Sentinel: design

## 1. Problem and requirements

A card issuer approves or declines each payment while the shop waits. Fraud costs money directly;
blocking a good customer costs money and trust; every manual review costs analyst time.

**Functional**
- Score every payment and return approve / review / block with reasons.
- Detect coordinated fraud (mule rings) that looks normal one payment at a time.
- Give analysts a queue with a readable note per case and a way to label it.
- Know when the model or its inputs change, and evaluate candidate models on live traffic safely.

**Non-functional**
- Latency: p99 under 50 ms server-side for the scoring call (card-network budgets are ~100 ms end to end).
- Payments must never wait on, or fail because of, anything that isn't needed for the decision
  (notes, graph, monitoring, LLMs).
- No full card numbers or personal data outside the scoring path; personal attributes never shown
  to analysts as a reason.
- Every live model traceable to its data, code settings, metrics and decision policy.

## 2. Architecture

See the diagram in the [README](../README.md#architecture).

**Request path** (`POST /score`, ~1 ms of work):
1. Validate the payment (Pydantic: category whitelist, ranges, timezone normalised to UTC).
2. One Redis Lua call: read the card's history (1h / 24h / 7d windows, lifetime stats, last
   location) and append this payment, atomically.
3. Build the feature vector in NumPy (no pandas on the hot path), LightGBM predict.
4. Apply the cost-based policy; for review/block, compute SHAP reasons.
5. Device rule: a device used by 5+ cards in 30 days is never auto-approved.
6. Publish the decision event (with the features the model saw) to Kafka, non-blocking; respond.

**Asynchronous consumers of the `decisions` topic**, each its own consumer group:

| Service | Does | State |
|---|---|---|
| `cases` | stores decisions, writes case notes (thread pool), stores ring alerts | SQLite (WAL) |
| `rings` | card-device graph, union-find, publishes `ring-alerts` | in memory |
| `modelops` | challenger in shadow, PSI drift every 1,000 events over the last 5,000 | SQLite |
| `dashboard` | reads the case store | none |

**Offline:** `data` (clean, drop PII) -> `features` (same `CardState` code as the online path) ->
`train` (time split, MLflow run + registry) -> `decisions` (policy, logged into the same run) ->
`challenger`.

## 3. Decisions and trade-offs

| Decision | Alternatives | Why |
|---|---|---|
| **Time-based split** (train < 2020-04-01 <= valid < test) | random split | Random splits leak the future (a card's later payments inform earlier ones) and overstate PR-AUC. |
| **LightGBM** on 17 features | deep/sequence models | Tabular, ~1.3M rows, needs per-payment explanations and sub-ms inference. Sequence information is captured by window features. |
| **Cost-minimising policy** with a review-capacity constraint | fixed 0.5 threshold, max-F1 | The business pays in dollars and analyst hours, not F1. Grid search over (review, block) lines on validation. |
| **Policy bound to model hash** | lines in config | A retrain changed the best block line from 0.69 to 0.85; stale lines would silently misbehave. API refuses a mismatched policy. |
| **Redis + Lua** for card state | Python dict, read-then-write | Shared across API replicas, survives restarts; Lua makes read+append atomic (two concurrent payments on one card both see each other). |
| **Kafka keyed by card**, at-least-once | direct DB writes from the API, exactly-once | Decouples the payment path. Per-card ordering is what consumers need. Duplicates are handled by idempotent writes (`txn_id` primary key) instead of transactions. |
| **Single uvicorn worker per container** | `--workers N` | Multi-worker uvicorn created sockets without `TCP_NODELAY` -> 40 ms delayed-ACK stalls (see incidents). Scale with replicas. |
| **Union-find** for rings | graph DB, GNN | Connected components incrementally, near O(1) per edge, small-to-large merging of member sets. Enough for card-device links; explainable to analysts. |
| **LLM notes behind a guard**, template fallback | LLM only, no LLM | Notes are off the payment path, so an LLM's latency is acceptable; its errors are not. Every number/date/time/name is checked against the facts. |
| **Shadow mode** before any rollout | offline comparison only | Offline wins don't guarantee live behaviour; shadow costs nothing for customers. |
| **PSI** with ok/watch/alert at 0.1/0.25 | KS tests | Bucketed, stable on large samples, standard in credit risk; also applied to the model score. |
| **SQLite** case store | PostgreSQL | Zero-ops for a single-node demo; WAL lets the dashboard read while the service writes. Schema migrations run on start. |

## 4. Features

All computed from **earlier** payments only (the current payment is added after its features are
read): counts and sums over 1h / 24h / 7d sliding windows, amount vs the card's average, seconds
and travel speed since the previous payment, distance to home, hour / day / night, category, plus
two cardholder attributes (age, city population; see the model card on fairness).

The sliding window is a deque with a running sum; when it empties, the sum is reset to exactly 0
(float residue like 1e-13 put live values in different PSI buckets than training).

Training-serving parity is tested: Redis vs training logic on random payments in CI, and checked on
50,000 real payments (0 mismatches); NumPy fast path vs pandas path on 3,000 payments (tolerance 1e-12).

## 5. Failure modes

| Failure | Effect | Handling |
|---|---|---|
| Kafka slow or down | events can't be sent | Producer is non-blocking; on a full local queue the event is dropped and counted. Payments are unaffected. |
| A consumer crashes | lag | Restarts from its last committed offset; `cases` commits only after rows are saved. Redelivered events are ignored by `txn_id`. |
| LLM timeout / error / invented fact | no LLM note | Template note, with the reason recorded (`note_rejected`). |
| Redis unavailable | no card history | `/health` returns 503 so a load balancer stops routing; scoring requests fail. **Gap:** a fallback that scores with request-only features (and flags it) is not built yet. |
| Two ring detectors in one group | each sees part of the graph | Detector checks its partition assignment on every rebalance and logs an error if partial. |
| Ring detector restart | graph memory lost | **Gap:** state isn't persisted; it should be rebuilt from the topic (`--from-start`) or kept in a shared store. |
| Model and policy mismatch | wrong decision lines | API refuses to start. |

## 6. Incidents

Each was found by running the system, root-caused with evidence, and fixed with a test where possible.

1. **Mixed timezones crashed the API; unknown categories were silently accepted.** Found by manual
   testing. Fix: normalise to naive UTC, category whitelist, range checks.
2. **40 ms per request in Docker.** Server time 1.3 ms, round trip 44 ms, but new connections were
   fast. Reproduced outside Docker by changing only the worker count. Root cause: with several
   workers uvicorn creates the socket with `proto=0`, asyncio then skips `TCP_NODELAY`, and Nagle +
   delayed ACK add ~40 ms. Fix: one worker per container. p50 44 -> 2.3 ms.
   ([details](PERFORMANCE.md#investigation-48-ms-round-trips-in-docker))
3. **Drift alarms on normal traffic.** Four separate causes: float residue in empty windows
   (`sum_1h`), rounded scores in events (`model_score`), cyclic features in a window shorter than
   their cycle (`day_of_week`), and a 3-day history backfill for 7-day features (`count_7d`,
   `sum_7d`, PSI 0.69 -> 0.07 with a 7-day backfill). After the fixes: 0 alerts on normal traffic.
4. **1 of 5 rings alerted live, 5 of 5 offline.** Ruled out the algorithm (shuffled replays), a
   crash (logs), lost events (read the whole topic), and lag (consumer groups: LAG 0). The group had
   two members: the service and a stray local process, each assigned 3 of 6 partitions. Events are
   keyed by card, so each saw half of every ring. Fix: stop the stray copy; assignment check.
5. **Drift check crashed on mixed timestamp formats** (with and without microseconds). Fix: ISO8601
   parsing; regression test.

## 7. Scaling

| Component | At 10-100x |
|---|---|
| API | Stateless replicas behind a load balancer (one process each); autoscale on p99. |
| Card state | Redis Cluster. The Lua script uses two keys per card (`recent`, `stats`); in a cluster they need a hash tag (`card:{id}:...`) to land in the same slot. Device keys are a separate call. |
| Kafka | More partitions; `cases` and `modelops` scale with consumers in their group. |
| Ring graph | Not by card partition (incident 4). Options: a graph store, Redis keyed by device with component ids, or repartitioning events by connected component. |
| Case store | PostgreSQL; notes written by a worker pool or a separate queue. |
| Training | Features rebuilt in Spark from the event log; scheduled retrain + policy selection (Airflow), promotion gated on shadow results. |

## 8. Next steps

- **One case per card** (or auto-freeze after N blocks): a stolen card in the demo produced 11 open
  cases in one night.
- **Label delay:** evaluate and retrain on labels as they arrive; monitor score drift in between.
- **Anomaly score as a feature** of the champion instead of a standalone challenger.
- **Fairness review** of `age` and `city_pop` (see the model card) before any real use.
- **Gradual rollout** (canary / A-B) after shadow, with automatic rollback on cost or latency.
- **Earlier ring alerts:** score components by risk instead of waiting for 5 cards (17% of ring
  money came after the alert).
