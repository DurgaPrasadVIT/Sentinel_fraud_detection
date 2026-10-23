"""Case notes for fraud analysts, written from the model's facts.

For every payment sent to review or blocked, write a short note an analyst can
read in 10 seconds: what happened, why it looks risky, what to do next.

Design rules:
- NEVER on the payment path: this service reads decisions from Kafka, so a slow
  or broken LLM can't slow down or break payments.
- The LLM only rephrases FACTS we give it. Every number in its note is checked
  against those facts; if it invents one, we use the template note instead.
- Privacy: the LLM only sees a masked card number (****1234), never the full one.
- Works without any LLM: the template writer is the default and the fallback.

Choose the writer with SENTINEL_LLM = template (default) | ollama | anthropic | openai
and SENTINEL_LLM_MODEL (e.g. llama3.1:8b for Ollama).
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def mask_card(card_id) -> str:
    return "****" + str(card_id)[-4:]


def money(x: float) -> str:
    return f"${x:,.2f}"


# Clues about WHO the cardholder is, not WHAT happened. The model uses them, but showing
# "risky because the cardholder is 61" to an analyst invites biased decisions. They are left
# out of notes; their use in the model is flagged for a fairness review (docs/LEARNING.md).
PERSONAL_CLUES = {"age", "city_pop"}


# How to say each clue in plain English. Only clues the model named as reasons are used.
def describe(name: str, clues: dict) -> str | None:
    v = clues.get(name)
    if v is None or name in PERSONAL_CLUES:
        return None
    return {
        "amount": lambda: f"amount {money(v)}",
        "sum_1h": lambda: f"{money(v)} spent on this card in the previous hour",
        "sum_24h": lambda: f"{money(v)} spent on this card in the previous 24 hours",
        "sum_7d": lambda: f"{money(v)} spent on this card in the previous 7 days",
        "count_1h": lambda: f"{v:.0f} earlier payments in the previous hour",
        "count_24h": lambda: f"{v:.0f} earlier payments in the previous 24 hours",
        "count_7d": lambda: f"{v:.0f} earlier payments in the previous 7 days",
        "amount_vs_avg": lambda: (f"amount is {v:.1f}x this card's average payment" if v >= 0.1
                                  else "amount is under a tenth of this card's average payment"),
        "secs_since_last": lambda: f"{v / 60:.0f} minutes after the previous payment",
        "speed_kmh": lambda: f"would require travelling at {v:,.0f} km/h from the previous shop",
        "hour": lambda: f"made in the {v:02.0f}:00 hour",
        "is_night": lambda: "made at night" if v else None,
        "category": lambda: f"category {v}",
        "distance_km": lambda: f"shop is {v:,.0f} km from the cardholder's home",
        "day_of_week": lambda: f"on a {DAY_NAMES[int(v)]}",
    }.get(name, lambda: None)()


@dataclass
class Facts:
    """Everything a note may say. Built only from the decision event."""

    card: str
    decision: str
    risk_pct: float
    amount: float
    category: str
    when: str
    reasons: list[str]

    @classmethod
    def from_event(cls, event: dict) -> "Facts":
        clues = event.get("clues") or {}
        reasons = []
        for name in event.get("reasons", []):
            device_rule = re.fullmatch(r"device_used_by_(\d+)_cards", name)
            if device_rule:
                reasons.append(f"device used by {device_rule.group(1)} different cards in 30 days")
            elif (text := describe(name, clues)) is not None:
                reasons.append(text)
        when = datetime.fromisoformat(str(event["ts"])).strftime("%Y-%m-%d %H:%M")
        return cls(card=mask_card(event["card_id"]), decision=event["decision"],
                   risk_pct=round(event["score"] * 100, 1), amount=float(event["amount"]),
                   category=str(event["category"]), when=when, reasons=reasons)

    def lines(self) -> list[str]:
        return [
            f"Card: {self.card}",
            f"Decision: {self.decision} (model risk {self.risk_pct}%)",
            f"Payment: {money(self.amount)}, category {self.category}, at {self.when}",
            *[f"Reason: {r}" for r in self.reasons],
        ]


NEXT_STEP = {
    "block": "Contact the cardholder; if they don't recognise it, close and reissue the card.",
    "review": "Confirm with the cardholder (call or app notification) before releasing the payment.",
}


def template_note(f: Facts) -> str:
    """Plain, reliable note. No AI involved."""
    reasons = "; ".join(f.reasons) if f.reasons else "overall pattern scored as risky"
    return (f"{f.decision.upper()} (risk {f.risk_pct}%): {money(f.amount)} {f.category} on card "
            f"{f.card} at {f.when}. Why: {reasons}. Next step: {NEXT_STEP.get(f.decision, '')}")


# ---------- Hallucination guard ----------

_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Things we have NO data about. If a note mentions them, the LLM made them up.
UNSUPPORTED_CLAIMS = re.compile(
    r"\b(report(ed|s)?|chargebacks?|complain(t|ts|ed)|police|confirmed|admitted|"
    r"previous fraud|known fraudster|blacklist(ed)?|ip address|vpn)\b", re.IGNORECASE)

# Capitalised words a note may use. Any other capitalised word in the middle of a
# sentence (a city, a shop, a person's name) is not in our facts.
ALLOWED_CAPITALISED = {"Card", "Contact", "Confirm", "BLOCK", "REVIEW", "Block", "Review",
                       "Next", "Why", "Reissue", "Close", "Verify", "Call", "PM", "AM"}


def _numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in _NUMBER.findall(text)]


def _strip(pattern: re.Pattern, text: str) -> str:
    return pattern.sub(" ", text)


def check_note(note: str, facts: Facts) -> list[str]:
    """Problems with an LLM note. Empty list = OK.

    - every number must match a number in the facts (rounding allowed)
    - times (HH:MM) and dates must match the facts exactly, as a whole
    - no long digit strings (possible full card numbers)
    - no claims about things we have no data for, no unknown names/places
    """
    problems = []
    fact_text = "\n".join(facts.lines())
    if re.search(r"\d{8,}", note.replace(",", "")):
        problems.append("contains a long number (possible full card number)")

    fact_times = {(int(h), int(m)) for h, m in _TIME.findall(fact_text)}
    for h, m in _TIME.findall(note):
        if (int(h), int(m)) not in fact_times:
            problems.append(f"time {h}:{m} is not in the facts")
    fact_dates = set(_DATE.findall(fact_text))
    for d in _DATE.findall(note):
        if d not in fact_dates:
            problems.append(f"date {d} is not in the facts")

    # Remaining numbers: compare with the facts' numbers (times and dates removed on both sides).
    allowed = _numbers(_strip(_TIME, _strip(_DATE, fact_text)))
    for n in _numbers(_strip(_TIME, _strip(_DATE, note))):
        if not any(abs(n - a) <= max(0.51, 0.01 * abs(a)) for a in allowed):
            problems.append(f"number {n:g} is not in the facts")

    if claim := UNSUPPORTED_CLAIMS.search(note):
        problems.append(f"claims something we have no data for: '{claim.group(0)}'")
    fact_words = set(re.findall(r"[A-Za-z_]+", fact_text))
    for sentence in re.split(r"(?<=[.!?:;])\s+", note):
        for word in re.findall(r"\b[A-Z][a-z]+\b", sentence)[1:]:  # skip the first word
            if word not in ALLOWED_CAPITALISED and word not in fact_words:
                problems.append(f"name or place '{word}' is not in the facts")
    if not note.strip():
        problems.append("empty note")
    return problems


# ---------- Writers ----------

SYSTEM_PROMPT = (
    "You write case notes for bank fraud analysts. Write 2-3 short sentences: what happened, "
    "why it looks risky, and the next step. Use ONLY the facts given. Never invent numbers, "
    "names or details. Refer to the card only as given (masked). Plain text, no lists."
)


class Writer(Protocol):
    name: str

    def write(self, facts: Facts) -> str: ...


class TemplateWriter:
    name = "template"

    def write(self, facts: Facts) -> str:
        return template_note(facts)


def _prompt(facts: Facts) -> str:
    return ("Facts:\n" + "\n".join(facts.lines())
            + f"\nSuggested next step: {NEXT_STEP.get(facts.decision, '')}\n\nCase note:")


class OllamaWriter:
    """Local open model (e.g. Llama 3.1 8B) on your own GPU. Data stays on the machine."""

    name = "ollama"

    def __init__(self, model: str, url: str = "http://localhost:11434", client=None) -> None:
        import httpx2 as httpx

        self.model, self.url = model, url
        self.client = client or httpx.Client(timeout=30)

    def write(self, facts: Facts) -> str:
        r = self.client.post(f"{self.url}/api/generate", json={
            "model": self.model, "system": SYSTEM_PROMPT, "prompt": _prompt(facts),
            "stream": False, "options": {"temperature": 0.2},
        })
        r.raise_for_status()
        return r.json()["response"].strip()


class AnthropicWriter:
    name = "anthropic"

    def __init__(self, model: str, api_key: str, client=None) -> None:
        import httpx2 as httpx

        self.model, self.api_key = model, api_key
        self.client = client or httpx.Client(timeout=30)

    def write(self, facts: Facts) -> str:
        r = self.client.post("https://api.anthropic.com/v1/messages", headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01",
        }, json={
            "model": self.model, "max_tokens": 300, "temperature": 0.2, "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": _prompt(facts)}],
        })
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"]).strip()


class OpenAIWriter:
    """OpenAI, or any OpenAI-compatible server (vLLM, LM Studio...) via OPENAI_BASE_URL."""

    name = "openai"

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1",
                 client=None) -> None:
        import httpx2 as httpx

        self.model, self.api_key, self.base_url = model, api_key, base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=30)

    def write(self, facts: Facts) -> str:
        r = self.client.post(f"{self.base_url}/chat/completions",
                             headers={"Authorization": f"Bearer {self.api_key}"}, json={
            "model": self.model, "temperature": 0.2, "max_tokens": 300,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": _prompt(facts)}],
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def make_writer() -> Writer:
    kind = os.getenv("SENTINEL_LLM", "template").lower()
    model = os.getenv("SENTINEL_LLM_MODEL")
    if kind == "template":
        return TemplateWriter()
    if kind == "ollama":
        return OllamaWriter(model or "llama3.1:8b", os.getenv("OLLAMA_URL", "http://localhost:11434"))
    if kind == "anthropic":
        return AnthropicWriter(model or "claude-haiku-5-5", os.environ["ANTHROPIC_API_KEY"])
    if kind == "openai":
        if not model:
            raise ValueError("Set SENTINEL_LLM_MODEL to the OpenAI model name")
        return OpenAIWriter(model, os.environ["OPENAI_API_KEY"],
                            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    raise ValueError(f"Unknown SENTINEL_LLM={kind!r}")


@dataclass
class Note:
    text: str
    source: str            # which writer produced the final text
    rejected: list[str]    # why an LLM note was rejected (empty if accepted or template)


def write_note(event: dict, writer: Writer) -> Note:
    """LLM note if it's safe, otherwise the template note. Never raises."""
    facts = Facts.from_event(event)
    if isinstance(writer, TemplateWriter):
        return Note(template_note(facts), "template", [])
    try:
        text = writer.write(facts)
    except Exception as exc:  # timeout, server down, quota...
        return Note(template_note(facts), "template", [f"{writer.name} failed: {exc}"[:200]])
    problems = check_note(text, facts)
    if problems:
        return Note(template_note(facts), "template", problems)
    return Note(text, writer.name, [])


SAMPLE_EVENT = {
    "txn_id": "sample", "card_id": 4532015112830366, "ts": "2020-12-21T23:51:00",
    "amount": 850.0, "category": "shopping_net", "decision": "block", "score": 0.9731,
    "reasons": ["sum_24h", "amount_vs_avg", "is_night"],
    "clues": {"sum_24h": 1765.0, "amount_vs_avg": 12.34, "is_night": 1},
}


def main() -> None:
    """Try the configured writer on one sample case:  python -m sentinel.casenotes"""
    writer = make_writer()
    print(f"Writer: {writer.name}\nFacts given to it:")
    print("  " + "\n  ".join(Facts.from_event(SAMPLE_EVENT).lines()))
    note = write_note(SAMPLE_EVENT, writer)
    print(f"\nNote (by {note.source}):\n{note.text}")
    if note.rejected:
        print(f"\nLLM note was NOT used, because: {note.rejected}")


if __name__ == "__main__":
    main()
