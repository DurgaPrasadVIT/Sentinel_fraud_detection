"""Fraud-ring detection: link cards through the devices they use.

One stolen card is a fraud case. Ten stolen cards cashed out from the same two
phones is a RING (often a money-mule operation). Single-payment models judge
each payment alone, so small, normal-looking ring payments can slip through.
A graph sees the connections:

    card A -- device 1 -- card B -- device 2 -- card C

A, B and C are one group even though A and C never shared a device.
Finding these groups = finding CONNECTED COMPONENTS, done with union-find.

Run as a live service (reads the Kafka "decisions" topic):
    python -m sentinel.rings

Run exactly ONE copy. Kafka splits a topic's partitions between the copies in a
consumer group, and events are keyed by card, so a ring's cards land in different
partitions. Two copies would each see half the graph and miss most rings. The
service checks its partition assignment and logs an error if it only got a part.
"""

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field


class UnionFind:
    """Groups items that are connected. Near-constant time per operation.

    - find(x): which group is x in? (returns the group's "root" item)
    - union(a, b): a and b are connected, so merge their groups
    Two tricks keep it fast on millions of items:
    - path compression: after find(x), x points straight at its root
    - union by size: the smaller group joins the bigger one, so trees stay flat
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> str:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:  # union by size
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra


@dataclass
class Ring:
    """A suspicious group of cards linked through shared devices."""

    cards: set
    devices: set
    payments: int
    amount: float
    already_stopped: int  # payments the single-payment model already sent to review/block

    def describe(self) -> str:
        return (f"{len(self.cards)} cards via {len(self.devices)} devices | "
                f"{self.payments} payments, ${self.amount:,.2f} | "
                f"model alone stopped {self.already_stopped}/{self.payments}")


@dataclass
class RingDetector:
    """Builds the card-device graph from payment events and flags large groups.

    Normal life: one person has 1-2 devices; a household may share a tablet
    (2-3 cards). A group of `min_cards` or more cards is unusual enough to alert.
    """

    min_cards: int = 5
    graph: UnionFind = field(default_factory=UnionFind)
    members: dict = field(default_factory=dict)  # group root -> card nodes in that group
    events: dict = field(default_factory=lambda: defaultdict(list))  # card -> its events
    devices_of: dict = field(default_factory=lambda: defaultdict(set))  # card -> devices
    alerted: set = field(default_factory=set)  # group roots already reported

    def _link(self, card: str, device: str) -> str:
        """Connect a card and a device; keep each group's card list up to date."""
        if card not in self.graph.parent:
            self.graph.add(card)
            self.members[card] = {card}
        ra, rb = self.graph.find(card), self.graph.find(device)
        if ra == rb:
            return ra
        root = self.graph.union(ra, rb)
        other = rb if root == ra else ra
        # The bigger group always keeps its set (union by size), so each card is
        # moved at most log2(n) times in total: "small-to-large" merging.
        self.members.setdefault(root, set()).update(self.members.pop(other, set()))
        if other in self.alerted:
            self.alerted.add(root)
        return root

    def add(self, event: dict) -> Ring | None:
        """Add one payment. Returns a Ring the first time a group becomes suspicious."""
        device = event.get("device_id")
        if not device:
            return None
        card = f"card:{event['card_id']}"
        self.events[card].append(event)
        self.devices_of[card].add(f"device:{device}")
        root = self._link(card, f"device:{device}")
        if root in self.alerted or len(self.members.get(root, ())) < self.min_cards:
            return None
        self.alerted.add(root)
        return self.ring_of(root)

    def ring_of(self, root: str) -> Ring:
        cards = self.members.get(root, set())
        events = [e for c in cards for e in self.events[c]]
        return Ring(
            cards={c.split(":", 1)[1] for c in cards},
            devices=set().union(*(self.devices_of[c] for c in cards)),
            payments=len(events),
            amount=round(sum(e["amount"] for e in events), 2),
            already_stopped=sum(e.get("decision", "approve") != "approve" for e in events),
        )

    def rings(self) -> list[Ring]:
        """Every group that is currently big enough to be suspicious."""
        found = [self.ring_of(root) for root, cards in self.members.items()
                 if len(cards) >= self.min_cards]
        return sorted(found, key=lambda r: -len(r.cards))


def coverage_problem(assigned: list[int], total: int) -> str | None:
    """None if this copy reads every partition, otherwise an error message.

    The graph only works if one process sees every payment (see the module docstring).
    """
    missing = sorted(set(range(total)) - set(assigned))
    if not missing:
        return None
    return (f"ERROR: this ring detector reads {len(set(assigned))} of {total} partitions "
            f"(missing {missing}). Another copy is running in the same consumer group. "
            "Rings need the whole card-device graph: stop the other copy, or rings will be missed.")


def main() -> None:
    from confluent_kafka import Consumer, Producer

    from sentinel.events import DECISIONS_TOPIC, encode, ensure_topic

    parser = argparse.ArgumentParser(description="Detect fraud rings from the decisions stream.")
    parser.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    parser.add_argument("--min-cards", type=int, default=5)
    parser.add_argument("--from-start", action="store_true")
    args = parser.parse_args()

    # Create the topics we read before subscribing. Services start in any order (e.g. in
    # docker compose); a consumer that subscribes to a topic that doesn't exist yet only
    # notices it minutes later, and with "latest" it has missed everything sent meanwhile.
    ensure_topic(args.bootstrap, DECISIONS_TOPIC)
    ensure_topic(args.bootstrap, "ring-alerts", partitions=1)
    consumer = Consumer({"bootstrap.servers": args.bootstrap, "group.id": "sentinel-rings",
                         "auto.offset.reset": "earliest" if args.from_start else "latest"})
    producer = Producer({"bootstrap.servers": args.bootstrap})
    total = len(consumer.list_topics(DECISIONS_TOPIC, timeout=10).topics[DECISIONS_TOPIC].partitions)

    def on_assign(_consumer, partitions) -> None:
        problem = coverage_problem([p.partition for p in partitions], total)
        print(problem or f"Reading all {total} partitions of '{DECISIONS_TOPIC}'.", flush=True)

    consumer.subscribe([DECISIONS_TOPIC], on_assign=on_assign)
    detector = RingDetector(min_cards=args.min_cards)
    print(f"Watching '{DECISIONS_TOPIC}' for rings of {args.min_cards}+ cards (Ctrl + C to stop)...")
    try:
        while True:
            msg = consumer.poll(0.5)
            if msg is None or msg.error():
                continue
            ring = detector.add(json.loads(msg.value()))
            if ring:
                print(f"\nRING ALERT: {ring.describe()}\n  cards: {sorted(ring.cards)}"
                      f"\n  devices: {sorted(ring.devices)}")
                # Alerts go to their own topic, for a case-management system to pick up.
                producer.produce("ring-alerts", value=encode({
                    "cards": sorted(ring.cards), "devices": sorted(ring.devices),
                    "payments": ring.payments, "amount": ring.amount,
                }))
                producer.poll(0)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()
        producer.flush(5)


if __name__ == "__main__":
    main()
