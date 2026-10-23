"""Evaluation of the hallucination guard on realistic LLM notes.

Cases: 8 real blocked payments from a simulator run (tests/data/casenote_cases.json).
- HONEST notes: written by an LLM (Claude) from the facts, with natural rounding and
  wording. The guard must ACCEPT them; otherwise it silently falls back to the
  template every time.
- HALLUCINATED notes: typical LLM mistakes. The guard must REJECT them.
"""

import json
from pathlib import Path

import pytest

from sentinel.casenotes import Facts, check_note

CASES = json.loads((Path(__file__).parent / "data" / "casenote_cases.json").read_text())

HONEST = [
    "Card ****0820 was blocked at 100% model risk for a $780.52 misc_net payment at 22:32 on 2020-06-21. The large amount and late-evening timing drove the decision. Contact the cardholder; if they don't recognise it, close and reissue the card.",
    "Card ****5478 was blocked for a $326.94 grocery_pos payment at 01:25, after $4,822.58 had already been spent on the card in the previous 24 hours. The overnight timing on top of that spending spree is a strong takeover signal. Contact the cardholder and reissue the card if the activity isn't theirs.",
    "A small $21.92 personal_care payment on card ****9330 was blocked because $2,142.58 was spent on the card in the prior 24 hours and it happened at night. Small purchases after a spree are typical of fraudsters testing a card. Contact the cardholder; if they don't recognise it, close and reissue the card.",
    "Card ****0777 was blocked for a $12.34 gas_transport payment just after midnight, following $2,081 of spending in the previous 24 hours. This looks like a stolen card being tested with small fuel purchases. Contact the cardholder and reissue the card if needed.",
    "Card ****9330 was blocked for a $777.45 in-store payment at 03:55, with 4 earlier payments in the previous week. A large purchase in the middle of the night is unusual for this card. Contact the cardholder; if they don't recognise it, close and reissue the card.",
    "Card ****1125 was blocked for a $7.61 gas_transport payment at 08:18 after $2,103.15 had been spent in the previous 24 hours. Tiny fuel transactions after heavy spending are a common card-testing pattern. Contact the cardholder and reissue the card if they don't recognise it.",
    "Card ****2564 was blocked for a $920 misc_net payment at 21:48, which is about 5 times its average payment, with 6 earlier payments in the past week. Contact the cardholder to confirm; if they don't recognise it, close and reissue the card.",
    "Card ****8832 was blocked for a $7.67 travel payment at 23:56, far below its usual amount, after $2,068.05 was spent in the previous 24 hours. This pattern fits a card being tested after a spree. Contact the cardholder; if they don't recognise it, close and reissue the card.",
]

HALLUCINATED = [
    (1, "Card ****5478 was blocked for a $362.94 grocery payment after a $4,822.58 spree. Contact the cardholder."),
    (1, "Card ****5478 was blocked: $326.94 at 01:25 after $8,422.58 in 24 hours. Contact the cardholder."),
    (6, "Card ****2564 spent $920, about 12 times its average. Contact the cardholder."),
    (0, "Card ****0820 was blocked for $780.52 at 02:32. Contact the cardholder."),
    (3, "Card ****0777 made 3 small fuel payments in 10 minutes after a $2,081.15 spree. Reissue the card."),
    (4, "Card ****9330 had 2 previous chargebacks; it was blocked for $777.45 at 03:55. Reissue the card."),
    (2, "Card ****9330 was blocked for $21.92 in Paris after $2,142.58 in 24 hours. Contact the cardholder."),
    (2, "The cardholder already reported card ****9330 stolen; block the $21.92 payment. Reissue the card."),
    (5, "Card 4532015112831125 was blocked for $7.61 after $2,103.15. Contact the cardholder."),
    (0, "Card ****0820 was blocked for $780.52 on 2020-07-21 at 22:32. Contact the cardholder."),
]


@pytest.mark.parametrize("case, note", list(enumerate(HONEST)))
def test_honest_notes_are_accepted(case, note):
    assert check_note(note, Facts.from_event(CASES[case])) == []


@pytest.mark.parametrize("case, note", HALLUCINATED)
def test_hallucinated_notes_are_rejected(case, note):
    assert check_note(note, Facts.from_event(CASES[case])) != []
