"""Publish every scoring decision to Kafka, for everything that happens AFTER.

The shop is waiting for an answer, so scoring is a direct API call.
The slow follow-up jobs (case notes, dashboards, fraud-ring detection,
saving data for retraining) must not make the shop wait. So the API answers
first, drops a copy of the decision into Kafka, and moves on. Other services
read it from Kafka in their own time.

Two publishers with the same interface:
- NullPublisher: does nothing. Used when Kafka is not configured (tests, local dev).
- KafkaPublisher: sends to Kafka without blocking the request.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Protocol

log = logging.getLogger(__name__)

DECISIONS_TOPIC = "decisions"
PARTITIONS = 6  # up to 6 consumers in one group can share the work


class Publisher(Protocol):
    def publish(self, event: dict) -> None: ...
    def close(self) -> None: ...


class NullPublisher:
    """Used when Kafka is not configured: events go nowhere."""

    def publish(self, event: dict) -> None:
        pass

    def close(self) -> None:
        pass


def encode(event: dict) -> bytes:
    """Events travel as JSON. Datetimes become ISO strings."""
    return json.dumps(event, default=lambda v: v.isoformat(), separators=(",", ":")).encode()


class KafkaPublisher:
    """Sends events to Kafka in the background (never blocks the API request)."""

    def __init__(self, producer, topic: str = DECISIONS_TOPIC) -> None:
        self.producer = producer
        self.topic = topic
        self.failed = 0

    def _on_delivery(self, err, msg) -> None:
        if err is not None:  # runs later, on a background thread
            self.failed += 1
            log.warning("Kafka delivery failed: %s", err)

    def publish(self, event: dict) -> None:
        try:
            # Key = card id, so all events of one card land in the same partition,
            # in order. Consumers then see each card's story in sequence.
            self.producer.produce(
                self.topic,
                key=str(event["card_id"]).encode(),
                value=encode(event),
                on_delivery=self._on_delivery,
            )
        except BufferError:
            # Local queue full (Kafka slow or down). Never fail the payment for this.
            self.failed += 1
            log.warning("Kafka queue full, dropping event %s", event.get("txn_id"))
        self.producer.poll(0)  # trigger delivery callbacks, without waiting

    def close(self) -> None:
        self.producer.flush(5)  # give queued events up to 5 s to be sent


def ensure_topic(bootstrap: str, topic: str = DECISIONS_TOPIC, partitions: int = PARTITIONS) -> None:
    """Create the topic if it doesn't exist yet."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    if topic in admin.list_topics(timeout=10).topics:
        return
    futures = admin.create_topics([NewTopic(topic, num_partitions=partitions, replication_factor=1)])
    for future in futures.values():
        try:
            future.result()
        except Exception as exc:  # another API copy may have created it at the same moment
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                raise


def make_publisher() -> Publisher:
    """Use Kafka when KAFKA_BOOTSTRAP is set (e.g. localhost:9092), otherwise do nothing."""
    bootstrap = os.getenv("KAFKA_BOOTSTRAP")
    if not bootstrap:
        return NullPublisher()
    from confluent_kafka import Producer

    ensure_topic(bootstrap)
    producer = Producer({
        "bootstrap.servers": bootstrap,
        "linger.ms": 5,               # wait up to 5 ms to send events in small batches
        "acks": "all",                # Kafka confirms the event is safely stored
        "enable.idempotence": True,   # retries never create duplicates
        "queue.buffering.max.messages": 100_000,
    })
    return KafkaPublisher(producer)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)
