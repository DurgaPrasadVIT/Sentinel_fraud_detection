"""Tests for sentinel.rings."""

from sentinel.rings import RingDetector, UnionFind


def pay(card, device, amount=50.0, decision="approve"):
    return {"card_id": card, "device_id": device, "amount": amount, "decision": decision}


def test_union_find_groups_connected_items():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("c", "d")
    assert uf.find("a") == uf.find("b")
    assert uf.find("a") != uf.find("c")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("d")
    assert uf.size[uf.find("a")] == 4


def test_union_find_stays_flat_on_long_chains():
    uf = UnionFind()
    for i in range(10_000):
        uf.union(f"n{i}", f"n{i + 1}")
    root = uf.find("n0")
    assert uf.parent["n0"] == root  # path compression: points straight at the root


def test_cards_linked_only_indirectly_are_one_ring():
    # 1 and 3 never share a device, but 2 connects them through devices x and y.
    detector = RingDetector(min_cards=3)
    detector.add(pay(1, "x"))
    detector.add(pay(2, "x"))
    ring = detector.add(pay(2, "y")) or detector.add(pay(3, "y"))
    assert ring is not None
    assert ring.cards == {"1", "2", "3"}
    assert ring.devices == {"device:x", "device:y"}


def test_household_sharing_a_tablet_is_not_a_ring():
    detector = RingDetector(min_cards=5)
    for card in (1, 2, 3):
        assert detector.add(pay(card, "family-tablet")) is None
    assert detector.rings() == []


def test_alerts_once_per_ring_and_counts_model_stops():
    detector = RingDetector(min_cards=3)
    alerts = [detector.add(pay(c, "mule", decision="review" if c == 1 else "approve"))
              for c in (1, 2, 3, 4)]
    assert sum(a is not None for a in alerts) == 1  # one alert, not one per payment
    ring = detector.rings()[0]
    assert ring.cards == {"1", "2", "3", "4"} and ring.already_stopped == 1


def test_payments_without_device_are_ignored():
    detector = RingDetector(min_cards=2)
    assert detector.add({"card_id": 1, "amount": 5.0}) is None
