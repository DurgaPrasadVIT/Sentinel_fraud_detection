"""Tests for sentinel.rings."""

from sentinel.rings import RingDetector, UnionFind, coverage_problem


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


def test_a_ring_split_across_two_detectors_is_missed():
    # Why only one copy may run: events are partitioned by card, so two copies in one
    # consumer group each see half of the ring's cards and neither reaches the limit.
    ring = [{"card_id": c, "device_id": f"mule-{c % 2}", "amount": 50.0} for c in range(6)]
    ring += [{"card_id": 0, "device_id": "mule-1", "amount": 50.0}]  # links the two devices
    whole = RingDetector(min_cards=5)
    assert [r for e in ring if (r := whole.add(e))]
    halves = [RingDetector(min_cards=5), RingDetector(min_cards=5)]
    alerts = [r for e in ring if (r := halves[e["card_id"] % 2].add(e))]
    assert alerts == []


def test_coverage_problem_flags_a_partial_assignment():
    assert coverage_problem([0, 1, 2, 3, 4, 5], 6) is None
    msg = coverage_problem([0, 1, 2], 6)
    assert "3 of 6" in msg and "[3, 4, 5]" in msg
