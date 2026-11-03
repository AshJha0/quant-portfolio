"""Queue-aware fills: hand cases, bounds, and a cross-check against a FIFO book.

The FIFO book below is a stand-in for the W25 reference book (price-time
priority, one level is enough for queue position). In it our order is a
real order; the model sees only the public view. Where the model's cancel
assumption matches what really happened, its fills must equal the book's.
Integration re-runs ``test_matches_fifo_*`` against ``qp_exec.lob``.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict

import pytest
from fills import Level, QueueAwareFillModel, fill_marketable, prob_ahead_power


class FifoBook:
    """Minimal price-time book: per (side, price) an ordered dict id -> qty."""

    def __init__(self) -> None:
        self.q: dict[tuple[str, float], OrderedDict[int, float]] = {}
        self.next_id = 1000

    def add(self, side: str, price: float, qty: float) -> int:
        self.next_id += 1
        self.q.setdefault((side, price), OrderedDict())[self.next_id] = qty
        return self.next_id

    def cancel(self, side: str, price: float, oid: int) -> None:
        del self.q[(side, price)][oid]

    def trade(self, side: str, price: float, qty: float) -> dict[int, float]:
        """Aggressor takes ``qty`` from resting ``side`` at ``price``."""
        done: dict[int, float] = {}
        lvl = self.q[(side, price)]
        for oid in list(lvl):
            if qty <= 0:
                break
            take = min(qty, lvl[oid])
            lvl[oid] -= take
            qty -= take
            done[oid] = take
            if lvl[oid] <= 0:
                del lvl[oid]
        return done

    # BookView, excluding a hidden "own" order id so the model sees the public depth
    hidden: int | None = None

    def depth_at(self, side, price):
        lvl = self.q.get((side, price), {})
        return float(sum(v for k, v in lvl.items() if k != self.hidden))

    def levels(self, side):
        prices = sorted({p for (s, p), lvl in self.q.items() if s == side and lvl}, reverse=(side == "buy"))
        return [Level(p, self.depth_at(side, p)) for p in prices]


def test_worked_example_from_docstring():
    book = FifoBook()
    book.add("buy", 99.0, 10)
    m = QueueAwareFillModel(book)
    o = m.submit("buy", 99.0, 3)
    assert o.queue_ahead == 10
    m.on_depth("buy", 99.0, 8)  # nobody behind: the cancel was in front
    assert o.queue_ahead == pytest.approx(8)
    f1 = m.on_trade(99.0, 9, aggressor="sell")
    assert [f.qty for f in f1] == [pytest.approx(1)]
    f2 = m.on_trade(98.5, 5, aggressor="sell")  # through our price
    assert [f.qty for f in f2] == [pytest.approx(2)]
    assert o.done


def test_no_fill_until_queue_ahead_consumed():
    book = FifoBook()
    book.add("sell", 101.0, 5)
    m = QueueAwareFillModel(book)
    o = m.submit("sell", 101.0, 2)
    assert m.on_trade(101.0, 5, aggressor="buy") == []
    assert o.queue_ahead == 0
    assert m.on_trade(100.0, 7, aggressor="buy") == []  # wrong side of us: no fill
    fills = m.on_trade(101.0, 1, aggressor="buy")
    assert fills[0].qty == 1 and o.remaining == 1


def test_cancel_bounds_and_pro_rata():
    for p, expect in ((0.0, 10.0), (1.0, 6.0), (prob_ahead_power, 10 - 4 * 10 / 15)):
        book = FifoBook()
        book.add("buy", 99.0, 10)
        m = QueueAwareFillModel(book, cancel_ahead=p)
        o = m.submit("buy", 99.0, 1)
        m._depth[("buy", 99.0)] = 15.0  # 5 joined behind us
        m.on_depth("buy", 99.0, 11.0)
        assert o.queue_ahead == pytest.approx(expect)


def test_trade_then_depth_not_double_counted():
    book = FifoBook()
    book.add("buy", 99.0, 10)
    m = QueueAwareFillModel(book, cancel_ahead=1.0)
    o = m.submit("buy", 99.0, 1)
    m.on_trade(99.0, 4, aggressor="sell")
    m.on_depth("buy", 99.0, 6.0)  # the feed reports the post-trade depth
    assert o.queue_ahead == pytest.approx(6.0)


def test_prob_ahead_power_limits():
    assert prob_ahead_power(0, 5) == 0.0
    assert prob_ahead_power(5, 0) == 1.0
    assert prob_ahead_power(3, 3, n=7) == pytest.approx(0.5)
    assert prob_ahead_power(1e6, 1, n=400) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        prob_ahead_power(-1, 1)


def test_submit_rejects_marketable_and_bad_input():
    book = FifoBook()
    book.add("sell", 100.0, 5)
    m = QueueAwareFillModel(book)
    with pytest.raises(ValueError, match="marketable"):
        m.submit("buy", 100.0, 1)
    with pytest.raises(ValueError):
        m.submit("buy", 99.0, 0)
    with pytest.raises(ValueError):
        m.submit("hold", 99.0, 1)  # type: ignore[arg-type]


def test_fill_marketable_walks_levels_and_reports_partial():
    book = FifoBook()
    book.add("sell", 100.0, 2)
    book.add("sell", 100.5, 3)
    book.add("sell", 101.0, 10)
    r = fill_marketable(book, "buy", 4)
    assert [(f.price, f.qty) for f in r.fills] == [(100.0, 2), (100.5, 2)]
    assert r.avg_price == pytest.approx((200 + 201) / 4)
    r2 = fill_marketable(book, "buy", 20, limit=100.5)
    assert r2.filled == 5 and r2.unfilled == 15
    r3 = fill_marketable(book, "sell", 1)
    assert r3.filled == 0 and math.isnan(r3.avg_price)


def _fifo_vs_model(seed: int, cancel_from: str) -> None:
    """Random flow on one bid level. Cancels come only from ``cancel_from``
    of our order; the model is told the same assumption, so fills match."""
    rng = random.Random(seed)
    book = FifoBook()
    for _ in range(rng.randint(1, 6)):
        book.add("buy", 99.0, rng.randint(1, 5))
    ours_qty = rng.randint(1, 8)
    ours = book.add("buy", 99.0, ours_qty)
    book.hidden = ours
    model = QueueAwareFillModel(book, cancel_ahead=1.0 if cancel_from == "front" else 0.0)
    # the model's order arrives with the same public depth in front
    book.hidden = ours
    vo = model.submit("buy", 99.0, ours_qty)
    true_filled = 0.0
    for _ in range(60):
        lvl = book.q[("buy", 99.0)]
        ids = list(lvl)
        pos = ids.index(ours) if ours in lvl else len(ids)
        r = rng.random()
        if r < 0.35:
            book.add("buy", 99.0, rng.randint(1, 4))
            model.on_depth("buy", 99.0, book.depth_at("buy", 99.0))
        elif r < 0.65:
            cands = ids[:pos] if cancel_from == "front" else ids[pos + 1:]
            if not cands:
                continue
            book.cancel("buy", 99.0, rng.choice(cands))
            model.on_depth("buy", 99.0, book.depth_at("buy", 99.0))
        else:
            total = sum(lvl.values())
            if total <= 0:
                continue
            q = rng.randint(1, max(1, int(total)))
            done = book.trade("buy", 99.0, q)
            true_filled += done.get(ours, 0.0)
            model.on_trade(99.0, q, aggressor="sell")
            model.on_depth("buy", 99.0, book.depth_at("buy", 99.0))
        assert vo.filled == pytest.approx(true_filled)
    assert vo.filled == pytest.approx(true_filled)


@pytest.mark.parametrize("seed", range(200))
def test_matches_fifo_when_cancels_behind(seed):
    _fifo_vs_model(seed, "behind")


@pytest.mark.parametrize("seed", range(200))
def test_matches_fifo_when_cancels_in_front(seed):
    _fifo_vs_model(seed, "front")
