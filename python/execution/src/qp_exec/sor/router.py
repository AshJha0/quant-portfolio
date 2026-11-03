"""The smart order router: sweep, allocate, post, learn.

Plain idea
----------
Every ``interval_us`` the router looks at what is left of the parent order,
estimates how much flow each venue will bring to a resting order (from what
it has seen there lately), prices each venue (half spread, fee or rebate,
and a learned adverse-selection charge), and solves the allocation LP of
``allocation.py``. The market part is sent as a sweep of IOC limit orders
across the lit books, best price first; the limit parts rest on lit books
at the touch and in dark venues at the mid. At the end of each interval
everything unfilled is cancelled and the loop repeats. At the horizon the
remainder is swept.

The sweep never takes a price worse than one it could see on another venue
at the time it decided (a trade-through check on its own snapshot). That is
a property of this simulator, not a claim of Regulation NMS Rule 611
compliance: venue quotes move during the latency, and protected-quote rules
have details not modelled here.

Anti-gaming randomisation (``randomise=True``): child sizes are drawn from
the same geometric distribution as background takers, the interval is
exponential around its mean, dark children carry a random minimum quantity
in ``min_qty_range``, and venue order is shuffled. Without it the router is
regular in size and time, which is what a predator's detector looks for.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from qp_exec.lob import Order, Side, Tif

from .allocation import solve_allocation
from .market import ExecFill, Market, sgn
from .tca import VenueScores


@dataclass(frozen=True)
class SweepLeg:
    venue: str
    price: int
    qty: int


def sweep_plan(mkt: Market, side: Side, qty: int, limit: int | None = None) -> list[SweepLeg]:
    """Split a marketable quantity over displayed lit depth, best price first.

    Ties at a price go to the lower taker fee, then the lower latency.
    Returns one IOC per (venue, price level) touched, priced at that level.
    """
    opp = Side.SELL if side == Side.BUY else Side.BUY
    levels = []
    for name, ven in mkt.lit.items():
        for px, q in ven.book.depth(opp, 20):
            if limit is not None and sgn(side) * (px - limit) > 0:
                break
            levels.append((sgn(side) * px, ven.sched.taker_ticks, ven.sched.latency_us, name, px, q))
    levels.sort()
    out: list[SweepLeg] = []
    left = qty
    for *_, name, px, q in levels:
        if left == 0:
            break
        take = min(left, q)
        out.append(SweepLeg(name, px, take))
        left -= take
    return out


@dataclass
class RouterReport:
    filled: int
    avg_price: float
    arrival_mid: float
    shortfall: float  # ticks per share incl. fees, positive = cost
    fees: float  # ticks per share
    sent: dict[str, int]
    fills: list[ExecFill]
    overfill: int


@dataclass
class SmartRouter:
    mkt: Market
    side: Side
    qty: int
    horizon_us: int
    interval_us: int = 200_000
    lam_u: float = 3.0  # ticks per share: what an unfilled lot costs at the horizon
    lam_floor: float = 0.2  # ... and at the start (urgency rises linearly between)
    dark: tuple[str, ...] = ()
    use_lit_passive: bool = True
    randomise: bool = False
    min_qty_range: tuple[int, int] = (2, 10)
    fixed_child: int = 5  # deterministic market-child size when not randomising
    scores: VenueScores | None = None
    window: int = 20
    seed: int = 0
    ioi: str | None = None  # None, "actionable" or "non-actionable": IOI each interval
    ioi_venue: str = "DARK_MID"
    decisions: list[dict] = field(default_factory=list)  # LP inputs/outputs, for audit
    _filled: int = 0
    _sent: dict[str, int] = field(default_factory=dict)
    _fills: list[ExecFill] = field(default_factory=list)
    _hist: dict[str, deque] = field(default_factory=dict)

    def _on_fill(self, f: ExecFill) -> None:
        self._filled += f.qty
        self._fills.append(f)
        if self.scores is not None:
            self.scores.observe(f)

    def _flow_snapshot(self) -> dict[str, int]:
        """Cumulative flow that would reach a resting order of ours, per venue."""
        snap = {}
        for n, v in self.mkt.lit.items():
            snap[n] = v.sell_taker_lots if self.side == Side.BUY else v.buy_taker_lots
        for n in self.dark:
            snap[n] = sum(q for _, q in self.mkt.dark[n].prints)
        return snap

    def _scenarios(self, venues: list[str], queue: dict[str, int], rng: np.random.Generator,
                   n: int = 200) -> np.ndarray:
        cols = []
        for v in venues:
            h = list(self._hist.get(v, ()))
            if not h:
                h = [0.0]
            xi = rng.choice(np.asarray(h, float), size=n)
            cols.append(np.maximum(xi - queue.get(v, 0), 0.0))
        return np.column_stack(cols) if cols else np.zeros((n, 0))

    def _send(self, venue: str, q: int) -> None:
        self._sent[venue] = self._sent.get(venue, 0) + q

    def _sweep(self, q: int, rng: np.random.Generator) -> None:
        legs = sweep_plan(self.mkt, self.side, q)
        if self.randomise:
            legs = [legs[i] for i in rng.permutation(len(legs))]
        for leg in legs:
            self.mkt.send_lit(leg.venue, Order(self.mkt.new_id(), self.side, leg.qty, leg.price,
                                               tif=Tif.IOC))
            self._send(leg.venue, leg.qty)

    def _max_latency(self) -> int:
        lat = [v.sched.latency_us for v in self.mkt.lit.values()]
        lat += [self.mkt.dark[n].sched.latency_us for n in self.dark]
        return max(lat)

    def run(self) -> RouterReport:
        m = self.mkt
        rng = np.random.default_rng(self.seed)
        m.listeners.append(self._on_fill)
        for n in self.dark:
            m.dark[n].firm_up = lambda o, x: min(x, max(0, self.qty - self._filled))
        t0 = m.t
        arrival = m.nbbo_mid()
        end = t0 + self.horizon_us
        last = self._flow_snapshot()
        for n in self.dark:
            # Cold-start prior for a dark venue (an assumption, replaced by
            # observed prints as the window fills): no contra in 70 % of
            # intervals, otherwise a lognormal block with median ~12 lots.
            pri = [0.0 if rng.random() < 0.7 else float(rng.lognormal(2.5, 0.8))
                   for _ in range(self.window)]
            self._hist.setdefault(n, deque(pri, maxlen=self.window))
        while m.t < end and self._filled < self.qty:
            if self.scores is not None:
                self.scores.update(m)
            need = self.qty - self._filled
            # Urgency: an unfilled lot costs little early (we can try again
            # next interval) and lam_u at the horizon.
            frac = min(1.0, (m.t - t0 + self.interval_us) / self.horizon_us)
            lam = max(self.lam_floor, self.lam_u * frac)
            b, a = m.nbbo()
            h = (a - b) / 2 if a is not None and b is not None else 0.5
            touch = b if self.side == Side.BUY else a
            venues, costs, queue = [], [], {}
            if self.use_lit_passive and touch is not None:
                for n, v in m.lit.items():
                    dep = v.book.depth(self.side, 1)
                    if dep and dep[0][0] == touch:
                        queue[n] = dep[0][1]
                        tox = self.scores.toxicity(n) if self.scores else 0.0
                        venues.append(n)
                        costs.append(-(h + v.sched.maker_ticks) + tox)
            for n in self.dark:
                tox = self.scores.toxicity(n) if self.scores else 0.0
                venues.append(n)
                costs.append(m.dark[n].sched.fee_ticks + tox)
            keep = [i for i, c in enumerate(costs) if c <= lam]
            venues = [venues[i] for i in keep]
            costs = [costs[i] for i in keep]
            taker = min(v.sched.taker_ticks for v in m.lit.values())
            D = self._scenarios(venues, queue, rng)
            if self.ioi is not None:
                m.ioi(self.side, need, self.ioi == "actionable", self.ioi_venue)
            if venues:
                alloc = solve_allocation(S=need, h=h, f=taker, c=costs, lam_u=lam, D=D)
                self.decisions.append({"S": need, "h": h, "f": taker, "c": list(costs),
                                       "lam_u": lam, "D": D, "venues": list(venues),
                                       "alloc": alloc})
                lim = [int(math.floor(x + 1e-9)) for x in alloc.limits]
                mq = int(round(alloc.market))
            else:
                lim, mq = [], need
            mq = min(mq, need - sum(lim))
            if not self.randomise and mq > 0:
                mq = min(mq, self.fixed_child)
            elif self.randomise and mq > 0:
                mq = min(mq, int(rng.geometric(1 / 3.0)))
            if mq > 0:
                self._sweep(mq, rng)
            working: list[tuple[str, int, bool]] = []
            for n, q in zip(venues, lim, strict=True):
                if q <= 0:
                    continue
                if n in m.lit:
                    oid = m.new_id()
                    m.send_lit(n, Order(oid, self.side, q, touch or 0))
                    working.append((n, oid, True))
                else:
                    lo, hi = self.min_qty_range
                    if self.randomise and q < lo:
                        continue  # a child below the min-qty floor is pingable
                    mn = int(rng.integers(lo, hi + 1)) if self.randomise else 0
                    oid = m.send_dark(n, self.side, q, min(mn, q))
                    working.append((n, oid, False))
                self._send(n, q)
            step = self.interval_us
            if self.randomise:
                step = max(m.p.step_us, int(rng.exponential(self.interval_us)))
            m.run_until(min(m.t + step, end))
            for n, oid, lit in working:
                if lit:
                    m.cancel_lit(n, oid)
                else:
                    m.cancel_dark(n, oid)
            # let cancels and in-flight IOCs land before the next decision,
            # so the no-oversubscription rule holds across intervals
            m.run_until(m.t + self._max_latency() + 1)
            now = self._flow_snapshot()
            for n in now:
                self._hist.setdefault(n, deque(maxlen=self.window)).append(now[n] - last.get(n, 0))
            last = now
        if self._filled < self.qty:
            m.run_until(m.t + max(v.sched.latency_us for v in m.lit.values()) + 1)
            left = self.qty - self._filled
            while left > 0:
                legs = sweep_plan(m, self.side, left)
                if not legs:
                    m.run_until(m.t + m.p.step_us)
                    continue
                self._sweep(left, rng)
                m.run_until(m.t + max(v.sched.latency_us for v in m.lit.values()) + 1)
                left = self.qty - self._filled
        m.run_until(m.t + max(d.sched.latency_us for d in m.dark.values()) + 1)
        m.listeners.remove(self._on_fill)
        q = sum(f.qty for f in self._fills)
        avg = sum(f.price * f.qty for f in self._fills) / q
        fees = sum(f.fee_ticks * f.qty for f in self._fills) / q
        sf = sgn(self.side) * (avg - arrival) + fees
        return RouterReport(q, avg, arrival, sf, fees, dict(self._sent), list(self._fills),
                            max(0, q - self.qty))
