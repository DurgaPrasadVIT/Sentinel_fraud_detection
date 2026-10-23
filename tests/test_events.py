"""Tests for sentinel.events."""

import json
from datetime import datetime

from sentinel.events import KafkaPublisher, NullPublisher, encode


class FakeProducer:
    """Records what would be sent to Kafka."""

    def __init__(self, full: bool = False) -> None:
        self.sent, self.full, self.flushed = [], full, False

    def produce(self, topic, key, value, on_delivery):
        if self.full:
            raise BufferError("queue full")
        self.sent.append((topic, key, value))

    def poll(self, timeout):
        return 0

    def flush(self, timeout):
        self.flushed = True


EVENT = {"txn_id": "t1", "card_id": 42, "ts": datetime(2020, 12, 21, 21, 0), "score": 0.9}


def test_encode_makes_json_with_iso_dates():
    assert json.loads(encode(EVENT))["ts"] == "2020-12-21T21:00:00"


def test_kafka_publisher_keys_by_card():
    producer = FakeProducer()
    KafkaPublisher(producer).publish(EVENT)
    topic, key, value = producer.sent[0]
    assert topic == "decisions"
    assert key == b"42"  # same card -> same partition -> events stay in order
    assert json.loads(value)["txn_id"] == "t1"


def test_full_queue_never_breaks_scoring():
    publisher = KafkaPublisher(FakeProducer(full=True))
    publisher.publish(EVENT)  # must not raise
    assert publisher.failed == 1


def test_close_flushes_waiting_events():
    producer = FakeProducer()
    KafkaPublisher(producer).close()
    assert producer.flushed


def test_null_publisher_does_nothing():
    NullPublisher().publish(EVENT)
    NullPublisher().close()
