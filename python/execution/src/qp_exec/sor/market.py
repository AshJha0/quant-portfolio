"""A multi-venue market: several lit price-time books and several dark venues.

Plain idea
----------
There is one "true" (efficient) price ``v`` that nobody sees exactly. It
drifts in regimes that informed traders know about, and every taker trade
leaves a permanent footprint on it. Each lit venue is a W25 reference book
(``qp_exec.lob.ReferenceBook``) quoted by its own market maker; uninformed
and informed takers arrive at random on each venue. Dark venues hold
non-displayed interest that trades at the lit midpoint. A router (ours)
talks to all of them with a per-venue latency.

Formal model (per time step of ``step_us`` microseconds)
--------------------------------------------------------
* Drift regime ``mu`` in {+mu0, -mu0}, switching with probability ``p_switch``.
* ``v <- v + mu + sigma * eps + kappa * (signed lit taker lots this step)``,
  ``eps ~ N(0, 1)``. Units: ticks.
* Lit venue ``k``: maker quotes ``levels`` price levels each side at
  ``floor(v) - i`` and ``floor(v) + 1 + i``, refreshed every step (orders at
  prices that are still wanted keep their queue position). Taker arrivals
  are Poisson with rate ``lit_rate`` per step per venue at uniformly random
  microsecond offsets; a fraction ``informed_share`` trade in the direction
  of ``mu``. Sizes are geometric with mean ``taker_mean`` lots.
* Dark venue arrivals: Poisson ``dark_rate`` per step per venue of
  counterparties of type uninformed, informed or predatory (see
  ``qp_exec.sor.dark``).

Everything is seeded, so an episode is reproducible bit for bit.

This is a stylised model calibrated to qualitative facts (dark fill rates
are low, informed flow makes passive fills adverse, dark midpoint trades
save the half spread), not to data: there is no free dark-pool data. Treat
every number it produces as an illustration of a mechanism.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qp_exec.lob import Fill, Order, OrdType, ReferenceBook, Side, Tif

from .fees import DEFAULT_DARK, DEFAULT_LIT, DarkSchedule, LitSchedule

ROUTER = "router"
MM = "mm"
BG = "bg"
INF = "informed"
PRED = "predator"


def sgn(side: Side) -> int:
    return 1 if side == Side.BUY else -1


@dataclass(frozen=True)
class MarketParams:
    step_us: int = 10_000
    sigma: float = 0.15
    mu0: float = 0.02
    p_switch: float = 0.01
    kappa: float = 0.01
    levels: int = 5
    mm_size: tuple[int, int] = (5, 15)
    lit_rate: float = 0.4
    informed_share: float = 0.3
    taker_mean: float = 3.0
    dark_rate: float = 0.05
    start_price: float = 10_000.5


@dataclass(frozen=True)
class ExecFill:
    """A fill of one of our (router) orders, wherever it happened."""

    venue: str
    ts: int
    side: Side
    price: float  # ticks; midpoint fills may be half ticks
    qty: int
    liquidity: str  # "A" aggressive (lit), "P" passive (lit), "D" dark
    fee_ticks: float  # per share, cost > 0, rebate < 0
    mid: float  # consolidated lit mid at the fill
    child_id: int
    counterparty: str


@dataclass(order=True)
class _Ev:
    ts: int
    seq: int
    fn: Callable[[], None] = field(compare=False)


class LitVenue:
    def __init__(self, sched: LitSchedule, mkt: Market) -> None:
        self.sched = sched
        self.name = sched.name
        self.book = ReferenceBook()
        self.mkt = mkt
        self.owner: dict[int, str] = {}
        self.mm_orders: dict[int, tuple[int, Side]] = {}  # price -> (order id, side)
        self.sell_taker_lots = 0  # flow that hits bids, for queue-depletion stats
        self.buy_taker_lots = 0
        self.tape: list[tuple[int, int, int]] = []  # (ts, +1 buy / -1 sell aggressor, qty)

    def submit(self, order: Order, who: str, ts: int) -> tuple[Fill, ...]:
        self.owner[order.order_id] = who
        res = self.book.add(order, ts)
        self._record(res.fills, ts)
        return res.fills

    def cancel(self, oid: int, ts: int) -> None:
        _, fills = self.book.cancel(oid, ts)
        self._record(fills, ts)

    def _record(self, fills: tuple[Fill, ...], ts: int) -> None:
        m = self.mkt
        for f in fills:
            if f.aggressor == Side.BUY:
                self.buy_taker_lots += f.qty
            elif f.aggressor == Side.SELL:
                self.sell_taker_lots += f.qty
            tk, mk = self.owner.get(f.taker_id, BG), self.owner.get(f.maker_id, BG)
            if f.aggressor is not None:
                self.tape.append((ts, sgn(f.aggressor), f.qty))
            if f.aggressor is not None:
                m.taker_flow += sgn(f.aggressor) * f.qty
            for who, oid, liq in ((tk, f.taker_id, "A"), (mk, f.maker_id, "P")):
                if who == ROUTER:
                    side = f.aggressor if liq == "A" else _opp(f.aggressor)
                    assert side is not None
                    fee = self.sched.taker_ticks if liq == "A" else -self.sched.maker_ticks
                    cp = mk if liq == "A" else tk
                    m.on_router_fill(ExecFill(self.name, ts, side, float(f.price), f.qty,
                                              liq, fee, m.nbbo_mid(), oid, cp))

    def requote(self, ts: int) -> None:
        m = self.mkt
        base = math.floor(m.v)
        want: dict[int, Side] = {}
        for i in range(m.p.levels):
            want[base - i] = Side.BUY
            want[base + 1 + i] = Side.SELL
        for px, (oid, side) in list(self.mm_orders.items()):
            st = self.book.order_state(oid)
            if st is None or want.get(px) != side:
                if st is not None:
                    self.cancel(oid, ts)
                del self.mm_orders[px]
        for px, side in want.items():
            if px in self.mm_orders:
                continue
            q = int(m.rng.integers(m.p.mm_size[0], m.p.mm_size[1] + 1))
            oid = m.new_id()
            self.submit(Order(oid, side, q, px), MM, ts)
            if self.book.order_state(oid) is not None:
                self.mm_orders[px] = (oid, side)

    def mid(self) -> float | None:
        b, a = self.book.best_bid(), self.book.best_ask()
        if b is None or a is None:
            return None
        return (b + a) / 2


def _void(_: object) -> None:
    return None


def _opp(s: Side | None) -> Side | None:
    if s is None:
        return None
    return Side.SELL if s == Side.BUY else Side.BUY


class Market:
    """The event loop. Our router reaches venues only through ``send_*``."""

    def __init__(self, seed: int, params: MarketParams | None = None,
                 lit: dict[str, LitSchedule] | None = None,
                 dark: dict[str, DarkSchedule] | None = None) -> None:
        from .dark import make_dark_venue  # local import: dark imports market

        self.p = params or MarketParams()
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.v = self.p.start_price
        self.mu = self.p.mu0
        self.taker_flow = 0
        self._ids = 0
        self._seq = 0
        self._heap: list[_Ev] = []
        self.lit = {k: LitVenue(s, self) for k, s in (lit or DEFAULT_LIT).items()}
        self.dark = {k: make_dark_venue(s, self) for k, s in (dark or DEFAULT_DARK).items()}
        self.fills: list[ExecFill] = []
        self.listeners: list[Callable[[ExecFill], None]] = []
        self.mid_t: list[int] = []
        self.mid_v: list[float] = []
        self.v_hist: list[float] = []
        self.predators: list = []
        for ven in self.lit.values():
            ven.requote(0)
        self._log_mid()

    # ------------------------------------------------------------ plumbing
    def new_id(self) -> int:
        self._ids += 1
        return self._ids

    def at(self, ts: int, fn: Callable[[], None]) -> None:
        self._seq += 1
        heapq.heappush(self._heap, _Ev(ts, self._seq, fn))

    def on_router_fill(self, f: ExecFill) -> None:
        self.fills.append(f)
        for cb in self.listeners:
            cb(f)

    def nbbo(self) -> tuple[int | None, int | None]:
        bids = [b for v in self.lit.values() if (b := v.book.best_bid()) is not None]
        asks = [a for v in self.lit.values() if (a := v.book.best_ask()) is not None]
        return (max(bids) if bids else None, min(asks) if asks else None)

    def nbbo_mid(self) -> float:
        b, a = self.nbbo()
        if b is None or a is None:
            return self.v
        return (b + a) / 2

    def _log_mid(self) -> None:
        self.mid_t.append(self.t)
        self.mid_v.append(self.nbbo_mid())
        self.v_hist.append(self.v)

    def mid_at(self, ts: int) -> float:
        """Consolidated mid as last logged at or before ``ts``."""
        i = int(np.searchsorted(self.mid_t, ts, side="right")) - 1
        return self.mid_v[max(i, 0)]

    # ------------------------------------------------------------ router API
    def send_lit(self, venue: str, order: Order) -> None:
        """Our order reaches ``venue`` after its latency."""
        ven = self.lit[venue]
        self.at(self.t + ven.sched.latency_us, lambda: _void(ven.submit(order, ROUTER, self.t)))

    def cancel_lit(self, venue: str, oid: int) -> None:
        ven = self.lit[venue]
        self.at(self.t + ven.sched.latency_us, lambda: ven.cancel(oid, self.t))

    def send_dark(self, venue: str, side: Side, qty: int, min_qty: int = 0) -> int:
        """Our dark order (firm, or conditional on a conditional venue)."""
        from .dark import DarkOrder

        dv = self.dark[venue]
        oid = self.new_id()
        self.at(self.t + dv.sched.latency_us,
                lambda: dv.submit(DarkOrder(oid, ROUTER, side, qty, min_qty, self.t)))
        return oid

    def cancel_dark(self, venue: str, oid: int) -> None:
        dv = self.dark[venue]
        self.at(self.t + dv.sched.latency_us, lambda: _void(dv.cancel(oid)))

    def ioi(self, side: Side, qty: int, actionable: bool, venue: str,
            p_resp: float = 0.3, p_leak_act: float = 0.2, p_leak_non: float = 0.5) -> bool:
        """Broadcast an indication of interest for ``qty`` on ``side``.

        Actionable IOIs carry a firm size and invite a contra to cross in
        ``venue``; non-actionable ones carry only the side and reach more
        recipients. Each predator learns the side with probability
        ``p_leak_act`` or ``p_leak_non`` (assumed values). Returns whether a
        contra responded.
        """
        from .dark import DarkOrder

        leak = p_leak_act if actionable else p_leak_non
        for pr in self.predators:
            if self.rng.random() < leak:
                pr.on_leak(side, "IOI")
        resp = p_resp if actionable else p_resp / 2
        if self.rng.random() >= resp:
            return False
        contra = Side.SELL if side == Side.BUY else Side.BUY
        q = min(qty, max(1, int(self.rng.lognormal(2.5, 0.8))))
        dv = self.dark[venue]
        self.at(self.t + dv.sched.latency_us + self.p.step_us // 2,
                lambda: dv.submit(DarkOrder(self.new_id(), BG, contra, q, 0, self.t,
                                            self.t + 5 * self.p.step_us)))
        return True

    # ------------------------------------------------------------ simulation
    def _background_step(self, t0: int) -> None:
        p = self.p
        for ven in self.lit.values():
            for _ in range(int(self.rng.poisson(p.lit_rate))):
                off = int(self.rng.integers(0, p.step_us))
                informed = self.rng.random() < p.informed_share
                if informed:
                    side = Side.BUY if self.mu > 0 else Side.SELL
                else:
                    side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
                q = int(self.rng.geometric(1.0 / p.taker_mean))
                who = INF if informed else BG

                def go(ven: LitVenue = ven, side: Side = side, q: int = q, who: str = who) -> None:
                    ven.submit(Order(self.new_id(), side, q, 0, OrdType.MARKET, Tif.IOC),
                               who, self.t)
                self.at(t0 + off, go)
        for dv in self.dark.values():
            dv.background(t0)
        for pr in self.predators:
            pr.step(t0)

    def _price_step(self) -> None:
        p = self.p
        if self.rng.random() < p.p_switch:
            self.mu = -self.mu
        self.v += self.mu + p.sigma * float(self.rng.standard_normal()) + p.kappa * self.taker_flow
        self.taker_flow = 0

    def run_until(self, t_end: int) -> None:
        """Advance the clock, processing every event with ``ts < t_end``."""
        step = self.p.step_us
        while self.t < t_end:
            boundary = (self.t // step + 1) * step
            stop = min(boundary, t_end)
            while self._heap and self._heap[0].ts < stop:
                ev = heapq.heappop(self._heap)
                self.t = max(self.t, ev.ts)
                ev.fn()
            self.t = stop
            if stop == boundary:
                self._price_step()
                for ven in self.lit.values():
                    ven.requote(self.t)
                for dv in self.dark.values():
                    dv.on_step(self.t)
                self._log_mid()
                self._background_step(self.t)
