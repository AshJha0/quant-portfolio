"""Dark venues, counterparties and the predatory trader.

Plain idea
----------
A dark pool shows nothing. Orders rest unseen and trade at the lit midpoint
when a contra order arrives. That saves the half spread, but you learn
nothing until you trade, the fill rate is low, and whoever trades with you
may know something you do not. A predator can "ping" a pool with tiny orders
to find large hidden interest and then trade ahead of it on the lit market.

Venue rules (all trades at the consolidated lit mid at the moment of match)
--------------------------------------------------------------------------
* ``MidpointPool`` - continuous crossing in time priority. A match of size
  ``x`` between orders ``o`` and ``c`` happens only if ``x >= min(o.min_qty,
  o.rem)`` and likewise for ``c``, where each ``min_qty`` is at least the
  venue floor. Minimum quantity therefore changes *which* contras we meet
  (only large ones), not just whether a fill is reported: pings and small
  informed slices can no longer reach us.
* ``ConditionalPool`` - every order is a conditional (a non-firm indication).
  When two conditionals match, each side is invited to firm up; the router
  decides for itself (through ``firm_up``), counterparties firm up with a
  type-dependent rate. A failed firm-up tells the other side that interest
  existed (leakage), and the venue suspends participants whose firm-up rate
  falls below ``firm_threshold`` after ``firm_min_invites`` invitations.
* ``PeriodicAuctionPool`` - frequent batch auction (Budish, Cramton and Shim
  2015): orders collect for ``interval_us`` and are matched at one price (the
  mid) at the end of the interval, in time priority, with the same min-qty
  rule. Speed inside the interval buys nothing.

Counterparties (assumed shares and rates, not estimates)
-------------------------------------------------------
``uninformed`` (random side, firm-up 0.9), ``informed`` (side of the drift,
firm-up 0.7; they also look at the lit book) and the ``Predator`` agent
(pings, firm-up 0.2, front-runs what it detects). Mittal (2008, "Are you
playing in a toxic dark pool?", Journal of Trading 3(3)) describes these
participant types and pinging; the numbers are ours.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from qp_exec.lob import Order, OrdType, Side, Tif

from .fees import DarkSchedule
from .market import BG, INF, PRED, ROUTER, ExecFill, Market, sgn

FIRM_RATE = {BG: 0.9, INF: 0.7, PRED: 0.2}


@dataclass
class DarkOrder:
    oid: int
    owner: str
    side: Side
    rem: int
    min_qty: int
    ts: int
    expire: int | None = None
    ioc: bool = False


@dataclass
class _Stats:
    invites: int = 0
    firms: int = 0


class MidpointPool:
    kind = "midpoint-continuous"

    def __init__(self, sched: DarkSchedule, mkt: Market) -> None:
        self.sched = sched
        self.name = sched.name
        self.mkt = mkt
        self.resting: list[DarkOrder] = []
        self.firm_stats: dict[str, _Stats] = {}
        self.leaks: list[tuple[int, str, Side]] = []  # (ts, who learned, side learned)
        self.prints: list[tuple[int, int]] = []  # (ts, qty): the venue's trade tape
        self.firm_up: Callable[[DarkOrder, int], int] | None = None  # router hook
        self.firm_threshold = 0.6
        self.informed_share: float | None = None  # per-venue override of the market's share
        self.rate_mult = 1.0  # per-venue multiplier on the market's dark arrival rate
        self.firm_min_invites = 5

    # --------------------------------------------------------------- orders
    def submit(self, o: DarkOrder) -> None:
        o.min_qty = max(o.min_qty, self.sched.min_qty)
        if self.suspended(o.owner):
            return
        self._cross(o)
        if o.rem > 0 and not o.ioc:
            self.resting.append(o)

    def cancel(self, oid: int) -> int:
        for i, o in enumerate(self.resting):
            if o.oid == oid:
                del self.resting[i]
                return o.rem
        return 0

    def suspended(self, who: str) -> bool:
        st = self.firm_stats.get(who)
        return (st is not None and st.invites >= self.firm_min_invites
                and st.firms / st.invites < self.firm_threshold)

    @staticmethod
    def _ok(o: DarkOrder, x: int) -> bool:
        return x >= min(o.min_qty, o.rem) and x > 0

    def _cross(self, o: DarkOrder) -> None:
        for c in list(self.resting):
            if o.rem == 0:
                break
            if c.side == o.side or c.rem == 0:
                continue
            x = min(o.rem, c.rem)
            if not (self._ok(o, x) and self._ok(c, x)):
                continue
            x = self._confirm(o, c, x)
            if x > 0:
                self._trade(o, c, x)
        self.resting = [c for c in self.resting if c.rem > 0]

    def _confirm(self, o: DarkOrder, c: DarkOrder, x: int) -> int:
        return x  # firm orders: no confirmation step

    def _trade(self, a: DarkOrder, b: DarkOrder, x: int) -> None:
        m = self.mkt
        px = m.nbbo_mid()
        a.rem -= x
        b.rem -= x
        self.prints.append((m.t, x))
        for me, other in ((a, b), (b, a)):
            if me.owner == ROUTER:
                m.on_router_fill(ExecFill(self.name, m.t, me.side, px, x, "D",
                                          self.sched.fee_ticks, px, me.oid, other.owner))
            elif me.owner == PRED:
                for pr in m.predators:
                    pr.on_ping_fill(self.name, me.side, x)

    # ----------------------------------------------------------- background
    def background(self, t0: int) -> None:
        m = self.mkt
        share = m.p.informed_share if self.informed_share is None else self.informed_share
        for _ in range(int(m.rng.poisson(m.p.dark_rate * self.rate_mult))):
            off = int(m.rng.integers(0, m.p.step_us))
            who = INF if m.rng.random() < share else BG
            if who == INF:
                side = Side.BUY if m.mu > 0 else Side.SELL
            else:
                side = Side.BUY if m.rng.random() < 0.5 else Side.SELL
            q = max(1, int(m.rng.lognormal(2.5, 0.8)))  # median ~12 lots
            life = int(m.rng.integers(5, 50)) * m.p.step_us
            mq = 0 if who == INF else int(m.rng.choice([0, 0, 5]))

            def go(side: Side = side, q: int = q, who: str = who, life: int = life,
                   mq: int = mq) -> None:
                self.submit(DarkOrder(m.new_id(), who, side, q, mq, m.t, m.t + life))
            m.at(t0 + off, go)

    def on_step(self, ts: int) -> None:
        self.resting = [o for o in self.resting if o.expire is None or o.expire > ts]


class ConditionalPool(MidpointPool):
    kind = "conditional"

    def _confirm(self, o: DarkOrder, c: DarkOrder, x: int) -> int:
        m = self.mkt
        out = x
        for me, other in ((o, c), (c, o)):
            st = self.firm_stats.setdefault(me.owner, _Stats())
            st.invites += 1
            if me.owner == ROUTER:
                got = self.firm_up(me, x) if self.firm_up else x
            else:
                got = x if m.rng.random() < FIRM_RATE.get(me.owner, 0.9) else 0
            if got > 0:
                st.firms += 1
                out = min(out, got)
            else:
                out = 0
            # The invitation itself tells each side the other's side exists.
            self.leaks.append((m.t, other.owner, me.side))
            if other.owner == PRED:
                for pr in m.predators:
                    pr.on_leak(me.side, self.name)
        if out == 0:
            return 0
        return out if self._ok(o, out) and self._ok(c, out) else 0


class PeriodicAuctionPool(MidpointPool):
    kind = "periodic-auction"

    def submit(self, o: DarkOrder) -> None:
        o.min_qty = max(o.min_qty, self.sched.min_qty)
        if not o.ioc:
            self.resting.append(o)

    def on_step(self, ts: int) -> None:
        iv = self.sched.interval_us or self.mkt.p.step_us
        if ts % iv == 0:
            self._uncross()
        super().on_step(ts)

    def _uncross(self) -> None:
        book = sorted(self.resting, key=lambda o: o.ts)
        for o in book:
            if o.rem == 0:
                continue
            for c in book:
                if o.rem == 0:
                    break
                if c is o or c.side == o.side or c.rem == 0:
                    continue
                x = min(o.rem, c.rem)
                if self._ok(o, x) and self._ok(c, x):
                    self._trade(o, c, x)
        self.resting = [c for c in self.resting if c.rem > 0]


def make_dark_venue(s: DarkSchedule, mkt: Market) -> MidpointPool:
    cls = {"midpoint-continuous": MidpointPool, "conditional": ConditionalPool,
           "periodic-auction": PeriodicAuctionPool}[s.model]
    return cls(s, mkt)


@dataclass
class Predator:
    """Pings dark pools with 1-lot IOCs; front-runs what it finds on lit.

    Each step it pings each continuous dark venue on both sides with
    probability ``ping_prob``. A filled ping on the sell side means someone
    is buying in the dark, so it buys ``front_qty`` lots on the lit venue
    with the best ask, and sells them back ``hold_steps`` steps later.
    ``detections`` keeps (ts, inferred side) for the detector features.
    """

    mkt: Market
    ping_prob: float = 0.3
    front_qty: int = 5
    hold_steps: int = 20
    fish_qty: int = 20
    detections: list[tuple[int, Side, str]] = field(default_factory=list)
    pings: int = 0
    pnl_ticks: float = 0.0

    def step(self, t0: int) -> None:
        m = self.mkt
        for dv in m.dark.values():
            if dv.kind == "conditional" and m.rng.random() < self.ping_prob:
                # fishing: a conditional it rarely firms up, to learn sides
                dv.submit(DarkOrder(m.new_id(), PRED, Side.BUY if m.rng.random() < 0.5
                                    else Side.SELL, self.fish_qty, 0, m.t,
                                    m.t + 5 * m.p.step_us))
                continue
            if dv.kind != "midpoint-continuous" or dv.sched.min_qty > 1:
                continue  # a 1-lot ping cannot meet a venue min-qty floor
            for side in (Side.BUY, Side.SELL):
                if m.rng.random() < self.ping_prob:
                    self.pings += 1
                    dv.submit(DarkOrder(m.new_id(), PRED, side, 1, 0, m.t, ioc=True))

    def on_ping_fill(self, venue: str, ping_side: Side, qty: int) -> None:
        # our ping sold -> a hidden buyer exists
        inferred = Side.BUY if ping_side == Side.SELL else Side.SELL
        self.detections.append((self.mkt.t, inferred, venue))
        self._front_run(inferred)

    def on_leak(self, side: Side, venue: str) -> None:
        self.detections.append((self.mkt.t, side, venue))
        self._front_run(side)

    def _front_run(self, side: Side) -> None:
        m = self.mkt
        ven = min(m.lit.values(), key=lambda v: v.sched.latency_us)
        start = m.nbbo_mid()
        ven.submit(Order(m.new_id(), side, self.front_qty, 0, OrdType.MARKET, Tif.IOC),
                   PRED, m.t)
        back = Side.SELL if side == Side.BUY else Side.BUY

        def unwind() -> None:
            ven.submit(Order(m.new_id(), back, self.front_qty, 0, OrdType.MARKET, Tif.IOC),
                       PRED, m.t)
            self.pnl_ticks += sgn(side) * (m.nbbo_mid() - start) * self.front_qty
        m.at(m.t + self.hold_steps * m.p.step_us, unwind)
