"""Tests for sentinel.decisions."""

import numpy as np

from sentinel.decisions import COST_FALSE_BLOCK, COST_REVIEW, Policy, money_lost


def test_decide_puts_scores_in_the_right_zone():
    policy = Policy(review_line=0.3, block_line=0.8)
    assert policy.decide(np.array([0.1, 0.5, 0.9])).tolist() == [0, 1, 2]


def test_money_lost():
    decisions = np.array([0, 2, 1, 0])   # approve, block, review, approve
    is_fraud = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 50.0, 70.0, 10.0])
    # Missed a $100 fraud + blocked one good customer + one review.
    assert money_lost(decisions, is_fraud, amount) == 100 + COST_FALSE_BLOCK + COST_REVIEW