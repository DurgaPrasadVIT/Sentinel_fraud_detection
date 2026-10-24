# Model card: `sentinel-fraud` v2

| | |
|---|---|
| Model | LightGBM binary classifier, 384 trees (early stopping on validation), 17 features |
| Registry | MLflow `sentinel-fraud`, version 2, run `fe5ddd56e62547b0be5f79cab10b5ed5` |
| Training data fingerprint | `e9d6891896b01c06` |
| Model file fingerprint | `dfa46d939561dedd` (the decision policy is bound to it) |
| Owner | Durgaprasad |

## Intended use

Real-time scoring of card-not-present and card-present payments to decide **approve / review /
block**, with the top reasons for each review or block. Decisions above the review line go to a
human or are blocked; the model never closes an account on its own.

**Not intended for:** credit decisions, judging individuals, or any data other than card
payments like the training data. Trained on synthetic data: not validated on real transactions.

## Data

[Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection),
generated with the Sparkov simulator: 1.85M payments, ~1,000 cards, 2019-01 to 2020-12.

| Split | Period | Payments | Fraud rate |
|---|---|---|---|
| Train | 2019-01-01 to 2020-03-31 | 1,097,693 | |
| Validation (early stopping, policy) | 2020-04-01 to 2020-06-21 | 198,982 | 0.58% (train + valid) |
| Test (reported) | 2020-06-21 to 2020-12-31 | 555,719 | 0.39% (2,145 fraud) |

Split by time, never at random. Names, street, job, gender and date of birth are dropped at
ingestion; age is derived from date of birth and kept (see fairness).

## Features

Card history over 1h / 24h / 7d (count, sum), amount vs the card's average, seconds and travel
speed since the previous payment, distance from home, hour, day of week, night flag, category,
amount, and two cardholder attributes: `age`, `city_pop`. All computed from earlier payments
only; the same code builds them offline and online (parity tested).

Importance (share of total gain): amount 37%, sum_24h 28%, category 8%, is_night 6%, age 3%,
hour 3%, city_pop 1.3%; the rest under 3% each.

## Performance (test period)

| Metric | Model | Amount-only baseline |
|---|---|---|
| PR-AUC | **0.969** (valid 0.982) | 0.137 |
| ROC-AUC | 0.999 | 0.833 |

**With the decision policy** (review if score >= 0.01, block if >= 0.79):

| | |
|---|---|
| Fraud dollars blocked or sent to review | 98.8% of $1,133,325 |
| Good customers blocked | 35 (block rate 0.33% of all payments) |
| Reviews per day | 4.6 (capacity 50) |
| Total cost | $17,043 |

The policy minimises `missed fraud $ + $20 x false blocks + $3 x reviews` on the validation
period, subject to at most 50 reviews per day. The cost figures are assumptions to be replaced
by the business's own.

Variance: adding `speed_kmh` lowered cost in 4 of 4 seeds (~12% on average); the released seed's
larger improvement is not the number to quote. v1 -> v2 (sliding-window float fix) cost $18,342
-> $17,043, within seed-to-seed noise.

## Fairness

`age` and `city_pop` describe **who** the cardholder is, not what happened. They carry ~4% of the
model's gain. They are **never shown to analysts** as a reason (notes omit them) to avoid biased
reviews. Outcomes by age on the test period:

| Age | Payments | Fraud caught (count) | Fraud $ caught | Good customers flagged per 10k | Blocked per 100k |
|---|---|---|---|---|---|
| < 30 | 94,017 | 99.7% | 100.0% | 13.7 | 9.6 |
| 30-44 | 185,867 | 95.5% | 98.3% | 12.9 | 5.9 |
| 45-59 | 152,764 | 94.5% | 97.7% | 13.5 | 7.2 |
| 60-74 | 80,283 | 97.0% | 99.7% | 9.3 | 5.0 |
| 75+ | 42,788 | 97.8% | 99.8% | 8.9 | 0.0 |

Good customers under 60 are flagged about 1.5x as often as those over 60. Block counts are
single digits per group, so those rates are not reliable. **Before any real use:** retrain without
`age` and `city_pop` and compare cost and these rates; keep them only if the gain is material and
the disparity is accepted by the business and compliance.

## Explanations

Top 3 SHAP contributions per review/block (LightGBM `pred_contrib`), translated into plain
sentences in case notes. Reasons are features that pushed the score up, not causes.

## Monitoring

- **Drift:** PSI per feature and on the score, last 5,000 events vs the validation snapshot
  (`models/reference.json`); ok < 0.1 <= watch < 0.25 <= alert. Cyclic features are skipped until
  the window covers a full cycle.
- **Shadow:** a challenger (Isolation Forest, PR-AUC 0.240, cost $397,947) scores live traffic
  without acting; agreement and volumes on the dashboard. Not promoted.
- **Retrain triggers:** sustained score drift, a rise in confirmed fraud missed, or cost above
  baseline on newly labelled data. After every retrain the policy is re-chosen and re-bound.

## Limitations

- Synthetic data: cards have no individual habits, no devices, no coordinated fraud. Real data
  will have lower scores and needs re-validation.
- Labels are immediate here; real chargebacks arrive weeks later.
- Synthetic fraud is concentrated in a few categories at night with large amounts, which the
  model relies on (amount, sum_24h, category, is_night carry ~80% of gain). Fraud that looks like
  normal spending (e.g. mule rings) is handled by the ring detector, not this model.
