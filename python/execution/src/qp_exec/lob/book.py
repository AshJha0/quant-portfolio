"""The slow reference limit order book.

This file is the executable form of ``lob/SPEC.md``. It is written for
clarity, not speed: every rule is a few lines that can be read against the
spec. The C++ fast book in ``cpp/exec-core`` must agree with it after every
event (the differential test in ``tests/test_differential_cpp.py``), and
other books in the repository (the W30 L3 book, the W27 venues) test against
it the same way.

Worked example
--------------
>>> from qp_exec.lob import ReferenceBook, Order, Side, Tif
>>> b = ReferenceBook()
>>> _ = b.add(Order(1, Side.SELL, 5, price=101), ts=1)
>>> _ = b.add(Order(2, Side.SELL, 5, price=102), ts=2)
>>> r = b.add(Order(3, Side.BUY, 7, price=102, tif=Tif.IOC), ts=3)
>>> [(f.maker_id, f.price, f.qty) for f in r.fills]
[(1, 101, 5), (2, 102, 2)]
>>> b.best_ask(), b.depth(Side.SELL, 1)
(102, [(102, 3)])
"""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field

from .types import AddResult, Fill, Order, OrdType, Peg, Phase, Reject, Side, Tif

_MASK = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def fnv_words(h: int, words: list[int]) -> int:
    """FNV-1a over 64-bit words (each word two's-complement masked to 64 bits)."""
    for w in words:
        h = ((h ^ (w & _MASK)) * _FNV_PRIME) & _MASK
    return h


@dataclass(slots=True)
class _Rest:
    oid: int
    side: Side
    price: int
    visible: int  # matchable now (the shown slice for icebergs)
    reserve: int  # iceberg reserve, 0 otherwise
    display: int  # iceberg slice size, 0 otherwise
    hidden: bool
    peg: Peg
    seq: int
    active: bool = True  # pegs are inactive while they cannot be priced

    @property
    def displayed_class(self) -> bool:
        return not self.hidden and self.peg != Peg.MID

    @property
    def total(self) -> int:
        return self.visible + self.reserve


@dataclass(slots=True)
class _Level:
    displayed: list[_Rest] = field(default_factory=list)
    hidden: list[_Rest] = field(default_factory=list)

    def empty(self) -> bool:
        return not self.displayed and not self.hidden

    def head(self) -> _Rest:
        return self.displayed[0] if self.displayed else self.hidden[0]

    def queue_of(self, o: _Rest) -> list[_Rest]:
        return self.displayed if o.displayed_class else self.hidden


class ReferenceBook:
    """Price-time priority book with pegs, icebergs, hidden orders and auctions.

    The published interface (other lanes may rely on these and nothing else):
    ``add``, ``cancel``, ``modify``, ``start_auction``, ``uncross``,
    ``best_bid``, ``best_ask``, ``depth``, ``order_state``, ``queue_ahead``,
    ``phase``, ``state_words`` and ``digest``.
    """

    def __init__(self) -> None:
        self._levels: dict[Side, dict[int, _Level]] = {Side.BUY: {}, Side.SELL: {}}
        # Ascending price lists; bids are read from the end.
        self._prices: dict[Side, list[int]] = {Side.BUY: [], Side.SELL: []}
        self._orders: dict[int, _Rest] = {}
        self._pegs: list[int] = []  # peg order ids in arrival order
        self._seq = 0
        self.phase = Phase.CONTINUOUS

    # ------------------------------------------------------------------ queries
    def _best(self, side: Side) -> int | None:
        p = self._prices[side]
        if not p:
            return None
        return p[-1] if side == Side.BUY else p[0]

    def _best_displayed(self, side: Side) -> int | None:
        ps = self._prices[side]
        it = reversed(ps) if side == Side.BUY else iter(ps)
        for px in it:
            if self._levels[side][px].displayed:
                return px
        return None

    def best_bid(self) -> int | None:
        """Best displayed bid (what a market-data subscriber sees)."""
        return self._best_displayed(Side.BUY)

    def best_ask(self) -> int | None:
        return self._best_displayed(Side.SELL)

    def depth(self, side: Side, levels: int) -> list[tuple[int, int]]:
        """Displayed L2 depth: ``[(price, shown qty), ...]`` from the best."""
        out: list[tuple[int, int]] = []
        ps = self._prices[side]
        it = reversed(ps) if side == Side.BUY else iter(ps)
        for px in it:
            lv = self._levels[side][px]
            q = sum(o.visible for o in lv.displayed)
            if q:
                out.append((px, q))
                if len(out) == levels:
                    break
        return out

    def order_state(self, order_id: int) -> tuple[int, int, int, bool] | None:
        """``(price, visible, reserve, active)`` of a resting order, else None."""
        o = self._orders.get(order_id)
        if o is None:
            return None
        return (o.price, o.visible, o.reserve, o.active)

    def queue_ahead(self, order_id: int) -> int | None:
        """Matchable quantity that trades before this order at its price.

        For a displayed order that is the shown quantity ahead of it in the
        displayed queue. For a non-displayed order it is every displayed lot
        at the level plus the hidden lots ahead of it. Iceberg reserves of
        orders ahead are not counted: they rejoin at the back when refilled.
        """
        o = self._orders.get(order_id)
        if o is None or not o.active:
            return None
        lv = self._levels[o.side][o.price]
        if o.displayed_class:
            ahead = 0
            for x in lv.displayed:
                if x is o:
                    return ahead
                ahead += x.visible
        ahead = sum(x.visible for x in lv.displayed)
        for x in lv.hidden:
            if x is o:
                return ahead
            ahead += x.visible
        raise AssertionError("order not in its level")  # pragma: no cover

    # -------------------------------------------------------------- internals
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _insert(self, o: _Rest) -> None:
        lv = self._levels[o.side].get(o.price)
        if lv is None:
            lv = _Level()
            self._levels[o.side][o.price] = lv
            insort(self._prices[o.side], o.price)
        lv.queue_of(o).append(o)

    def _unlink(self, o: _Rest) -> None:
        """Remove from its level (the order record itself is kept)."""
        lv = self._levels[o.side][o.price]
        lv.queue_of(o).remove(o)
        if lv.empty():
            del self._levels[o.side][o.price]
            self._prices[o.side].remove(o.price)

    def _drop(self, o: _Rest) -> None:
        if o.active:
            self._unlink(o)
        del self._orders[o.oid]
        if o.peg != Peg.NONE:
            self._pegs.remove(o.oid)

    def _consume(self, o: _Rest, q: int) -> None:
        """Take ``q`` lots from a resting order, refilling iceberg slices."""
        while q > 0:
            t = min(q, o.visible)
            o.visible -= t
            q -= t
            if o.visible == 0:
                if o.reserve == 0:
                    self._drop(o)
                    return
                # Refill: the new slice loses time priority (back of queue).
                self._unlink(o)
                o.visible = min(o.display, o.reserve)
                o.reserve -= o.visible
                o.seq = self._next_seq()
                self._insert(o)

    @staticmethod
    def _crosses(side: Side, limit: int | None, px: int) -> bool:
        if limit is None:
            return True
        return px <= limit if side == Side.BUY else px >= limit

    def _available(self, side: Side, limit: int | None, need: int) -> int:
        """Lots on the opposite side at acceptable prices, capped at ``need``."""
        opp = Side.SELL if side == Side.BUY else Side.BUY
        ps = self._prices[opp]
        it = iter(ps) if opp == Side.SELL else reversed(ps)
        tot = 0
        for px in it:
            if not self._crosses(side, limit, px):
                break
            lv = self._levels[opp][px]
            for o in lv.displayed:
                tot += o.total
            for o in lv.hidden:
                tot += o.total
            if tot >= need:
                return tot
        return tot

    def _match(self, oid: int, side: Side, limit: int | None, qty: int, ts: int,
               fills: list[Fill]) -> int:
        opp = Side.SELL if side == Side.BUY else Side.BUY
        rem = qty
        while rem > 0:
            px = self._best(opp)
            if px is None or not self._crosses(side, limit, px):
                break
            head = self._levels[opp][px].head()
            q = min(rem, head.visible)
            fills.append(Fill(oid, head.oid, px, q, ts, side))
            rem -= q
            self._consume(head, q)
        return rem

    def _peg_target(self, o: _Rest) -> int | None:
        bb = self._ref_best(Side.BUY)
        ba = self._ref_best(Side.SELL)
        if o.peg == Peg.PRIMARY:
            return bb if o.side == Side.BUY else ba
        if bb is None or ba is None:
            return None
        s = bb + ba
        return s // 2 if o.side == Side.BUY else -((-s) // 2)

    def _ref_best(self, side: Side) -> int | None:
        """Best price among non-pegged resting orders (displayed or hidden)."""
        ps = self._prices[side]
        it = reversed(ps) if side == Side.BUY else iter(ps)
        for px in it:
            lv = self._levels[side][px]
            for o in lv.displayed:
                if o.peg == Peg.NONE:
                    return px
            for o in lv.hidden:
                if o.peg == Peg.NONE:
                    return px
        return None

    def _after_event(self, ts: int, fills: list[Fill]) -> None:
        if self.phase != Phase.CONTINUOUS or not self._pegs:
            return
        # Reference prices are fixed before any peg moves: pegs never price
        # off each other.
        targets = [(oid, self._peg_target(self._orders[oid])) for oid in self._pegs]
        for oid, tgt in targets:
            o = self._orders[oid]
            if tgt is None:
                if o.active:
                    self._unlink(o)
                    o.active = False
            elif not o.active or o.price != tgt:
                if o.active:
                    self._unlink(o)
                o.price = tgt
                o.seq = self._next_seq()
                o.active = True
                self._insert(o)
        # Only two midpoint pegs at an integer midpoint can cross. The older
        # one is the maker.
        while True:
            bb = self._best(Side.BUY)
            ba = self._best(Side.SELL)
            if bb is None or ba is None or bb < ba:
                return
            b = self._levels[Side.BUY][bb].head()
            a = self._levels[Side.SELL][ba].head()
            maker, taker = (b, a) if b.seq < a.seq else (a, b)
            q = min(b.visible, a.visible)
            fills.append(Fill(taker.oid, maker.oid, maker.price, q, ts, taker.side))
            self._consume(b, q)
            self._consume(a, q)

    # ------------------------------------------------------------ public API
    @staticmethod
    def _valid(o: Order) -> bool:
        if o.qty <= 0 or o.display_qty < 0:
            return False
        if o.peg != Peg.NONE:
            return (o.ord_type == OrdType.LIMIT and o.tif == Tif.DAY and not o.post_only
                    and not o.hidden and o.display_qty == 0)
        if o.ord_type == OrdType.MARKET:
            return (o.tif != Tif.DAY and not o.post_only and not o.hidden
                    and o.display_qty == 0)
        if o.price <= 0:
            return False
        if o.hidden and o.display_qty:
            return False
        if o.post_only and o.tif != Tif.DAY:
            return False
        if (o.display_qty or o.hidden) and o.tif != Tif.DAY:
            return False
        return True

    def add(self, order: Order, ts: int) -> AddResult:
        """Submit a new order. Market orders must be IOC or FOK."""
        o = order
        if not self._valid(o):
            return AddResult(False, Reject.INVALID, (), 0)
        if o.order_id in self._orders:
            return AddResult(False, Reject.DUPLICATE_ID, (), 0)
        fills: list[Fill] = []
        if self.phase == Phase.AUCTION:
            if o.ord_type != OrdType.LIMIT or o.tif != Tif.DAY or o.post_only or o.peg:
                return AddResult(False, Reject.NOT_IN_AUCTION, (), 0)
            r = self._rest_new(o)
            return AddResult(True, Reject.NONE, (), r.total)

        if o.peg != Peg.NONE:
            r = _Rest(o.order_id, o.side, 0, o.qty, 0, 0, False, o.peg, 0, active=False)
            self._orders[o.order_id] = r
            self._pegs.append(o.order_id)
            self._after_event(ts, fills)
            live = self._orders.get(o.order_id)
            return AddResult(True, Reject.NONE, tuple(fills), live.total if live else 0)

        limit = None if o.ord_type == OrdType.MARKET else o.price
        opp = Side.SELL if o.side == Side.BUY else Side.BUY
        if o.post_only:
            px = self._best(opp)
            if px is not None and self._crosses(o.side, limit, px):
                return AddResult(False, Reject.POST_ONLY_WOULD_CROSS, (), 0)
        if o.tif == Tif.FOK and self._available(o.side, limit, o.qty) < o.qty:
            return AddResult(False, Reject.FOK_UNFILLABLE, (), 0)
        rem = self._match(o.order_id, o.side, limit, o.qty, ts, fills)
        resting = 0
        if rem > 0 and o.tif == Tif.DAY:
            r = self._rest_new(o, rem)
            resting = r.total
        self._after_event(ts, fills)
        live = self._orders.get(o.order_id)
        if live is None:
            resting = 0
        return AddResult(True, Reject.NONE, tuple(fills), resting)

    def _rest_new(self, o: Order, qty: int | None = None) -> _Rest:
        q = o.qty if qty is None else qty
        vis = min(o.display_qty, q) if o.display_qty else q
        r = _Rest(o.order_id, o.side, o.price, vis, q - vis, o.display_qty,
                  o.hidden, Peg.NONE, self._next_seq())
        self._orders[o.order_id] = r
        self._insert(r)
        return r

    def cancel(self, order_id: int, ts: int = 0) -> tuple[bool, tuple[Fill, ...]]:
        """Cancel; returns ``(found, fills)`` (fills come from peg repricing)."""
        o = self._orders.get(order_id)
        if o is None:
            return False, ()
        self._drop(o)
        fills: list[Fill] = []
        self._after_event(ts, fills)
        return True, tuple(fills)

    def modify(self, order_id: int, new_qty: int, ts: int = 0) -> tuple[bool, tuple[Fill, ...]]:
        """Set the remaining quantity. Down keeps priority; up loses it.

        A decrease comes out of the iceberg reserve first. An increase goes to
        the reserve of an iceberg and to the matchable quantity otherwise.
        ``new_qty <= 0`` cancels.
        """
        o = self._orders.get(order_id)
        if o is None:
            return False, ()
        if new_qty <= 0:
            return self.cancel(order_id, ts)
        if new_qty < o.total:
            cut = o.total - new_qty
            r = min(cut, o.reserve)
            o.reserve -= r
            o.visible -= cut - r
        elif new_qty > o.total:
            add = new_qty - o.total
            if o.display:
                o.reserve += add
            else:
                o.visible += add
            if o.active:
                self._unlink(o)
            o.seq = self._next_seq()
            if o.active:
                self._insert(o)
        fills: list[Fill] = []
        self._after_event(ts, fills)
        return True, tuple(fills)

    def start_auction(self) -> None:
        """Enter the call phase: orders rest without matching; pegs sit out."""
        self.phase = Phase.AUCTION
        for oid in self._pegs:
            o = self._orders[oid]
            if o.active:
                self._unlink(o)
                o.active = False

    def auction_price(self, ref_price: int) -> tuple[int | None, int]:
        """Uncross price and volume by the four standard rules.

        1. maximise executable volume; 2. minimise the absolute imbalance;
        3. closest to ``ref_price``; 4. the lower price.
        """
        cands = sorted(set(self._prices[Side.BUY]) | set(self._prices[Side.SELL]))
        best: tuple[int, int, int, int] | None = None
        best_px: int | None = None
        for p in cands:
            d = sum(o.total for px, lv in self._levels[Side.BUY].items() if px >= p
                    for o in lv.displayed + lv.hidden)
            s = sum(o.total for px, lv in self._levels[Side.SELL].items() if px <= p
                    for o in lv.displayed + lv.hidden)
            v = min(d, s)
            key = (-v, abs(d - s), abs(p - ref_price), p)
            if v > 0 and (best is None or key < best):
                best, best_px = key, p
        if best is None:
            return None, 0
        return best_px, -best[0]

    def uncross(self, ts: int, ref_price: int) -> tuple[Fill, ...]:
        """Run the auction at one price, then return to continuous trading."""
        px, vol = self.auction_price(ref_price)
        fills: list[Fill] = []
        if px is not None:
            def queue(side: Side) -> list[_Rest]:
                ps = self._prices[side]
                it = reversed(ps) if side == Side.BUY else iter(ps)
                out: list[_Rest] = []
                for p in it:
                    if (side == Side.BUY and p < px) or (side == Side.SELL and p > px):
                        break
                    lv = self._levels[side][p]
                    out.extend(lv.displayed)
                    out.extend(lv.hidden)
                return out

            buys, sells = queue(Side.BUY), queue(Side.SELL)
            brem = [b.total for b in buys]
            srem = [s.total for s in sells]
            i = j = 0
            left = vol
            while left > 0:
                q = min(brem[i], srem[j], left)
                fills.append(Fill(buys[i].oid, sells[j].oid, px, q, ts, None))
                brem[i] -= q
                srem[j] -= q
                left -= q
                if brem[i] == 0:
                    i += 1
                if srem[j] == 0:
                    j += 1
            for k, b in enumerate(buys):
                done = b.total - brem[k]
                if done:
                    self._consume(b, done)
            for k, s in enumerate(sells):
                done = s.total - srem[k]
                if done:
                    self._consume(s, done)
        self.phase = Phase.CONTINUOUS
        self._after_event(ts, fills)
        return tuple(fills)

    # --------------------------------------------------------- state digest
    def state_words(self) -> list[int]:
        """Canonical integer serialisation of the full book (both books hash it)."""
        w: list[int] = [int(self.phase)]
        for side in (Side.BUY, Side.SELL):
            ps = self._prices[side]
            it = reversed(ps) if side == Side.BUY else iter(ps)
            for px in it:
                lv = self._levels[side][px]
                w.append(px)
                for o in lv.displayed:
                    w += [o.oid, o.visible, o.reserve]
                w.append(-1)
                for o in lv.hidden:
                    w += [o.oid, o.visible]
                w.append(-2)
            w.append(-3)
        for oid in self._pegs:
            if not self._orders[oid].active:
                w.append(oid)
        w.append(-4)
        return w

    def digest(self, fills: tuple[Fill, ...] | list[Fill] = ()) -> int:
        words: list[int] = []
        for f in fills:
            words += [f.taker_id, f.maker_id, f.price, f.qty]
        words.append(-5)
        return fnv_words(_FNV_OFFSET, words + self.state_words())

    def check_invariants(self) -> None:
        """Raise AssertionError if the book is malformed (used by property tests)."""
        if self.phase == Phase.CONTINUOUS:
            bb, ba = self._best(Side.BUY), self._best(Side.SELL)
            assert bb is None or ba is None or bb < ba, f"crossed {bb} >= {ba}"
        seen = 0
        for side in (Side.BUY, Side.SELL):
            for px, lv in self._levels[side].items():
                assert not lv.empty()
                for o in lv.displayed + lv.hidden:
                    assert o.price == px and o.side == side and o.active
                    assert o.visible > 0 and o.reserve >= 0
                    assert self._orders[o.oid] is o
                    seen += 1
            assert self._prices[side] == sorted(self._levels[side])
        inactive = sum(1 for o in self._orders.values() if not o.active)
        assert seen + inactive == len(self._orders)

    def total_resting(self) -> int:
        return sum(o.total for o in self._orders.values())
