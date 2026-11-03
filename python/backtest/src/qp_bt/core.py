"""One backtest core: a shared clock, a typed event queue and two modes.

Plain idea
----------
Every strategy backtester in the repository does the same four things each
bar: carry the holdings to the fill, trade, pay for the trade, and carry the
new holdings to the close. This module does them once. *Event mode* holds
shares and cash and processes typed events from a queue; it is the reference.
*Fast mode* does the same arithmetic on weights in one numpy pass per bar for
weight-path strategies; it must agree with event mode to 1e-10 on every
fixture (``tests/test_fast_equals_event.py``).

Formal statement (one period ``s``, from close ``s-1`` to close ``s``)
--------------------------------------------------------------------
``h`` are the weights held at close ``s-1`` as fractions of ``NAV_{s-1}``;
``r`` the period's simple returns; ``a = phi r`` and
``b = (1 + r)/(1 + a) - 1`` the pre- and post-fill returns (``phi`` is the
fill fraction, see :mod:`qp_bt.fills`).

1. Carry, on what was held overnight: financing on the cash weight
   ``c = 1 - sum(h)`` (``qp_core.stats.Financing``: ``c (r_f -/+ spread)
   tau``) and borrow on the short book at the fill point,
   ``beta * sum(max(-h, 0)(1 + a))`` with ``beta`` the per-period borrow rate.
2. Pre-fill: ``m = 1 + h . a`` is NAV at the fill over ``NAV_{s-1}``; the
   holdings are worth ``H = h (1 + a)``.
3. Trade to the target ``w``, as weights of the post-trade NAV ``N``: the
   trade is ``dv = w N - H`` and ``N = m - C(dv)``, with ``C`` the cost model
   plus the fill's ``extra_cost_bp``. ``N`` is the fixed point of that
   equation (a contraction: costs are a small fraction of what is traded);
   it is solved to machine precision.
4. Post-fill: the book earns ``N (w . b)``.

Ledger (fractions of ``NAV_{s-1}``): ``cost = m - N``,
``gross = m - 1 + N (w . b)``, ``net = gross - cost - borrow + financing``,
``turnover = sum |dv|``, ``excess = net - r_f``; the next ``h`` is
``w N (1 + b) / (1 + net)``. With no order in the period, ``gross = h . r``.

Worked example
--------------
One asset, ``h = 0``, target ``w = 1``, 10 bp, fill fraction 0.5, ``r = 2%``:
``a = 0.01``, ``m = 1``, ``N = 1 - 0.001 N`` so ``N = 1/1.001``,
``cost = 0.000999``, ``b = 1.02/1.01 - 1``, ``gross = N b = 0.0098911``,
``net = 0.0088921``. ``tests/test_hand_fixtures.py`` checks this and a
3-asset, 10-day book against an exact rational re-computation in shares and
cash.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qp_core.stats import Financing

from .costs import ZERO_COST, CostModel
from .fills import NextBarFill

__all__ = [
    "BacktestLedger",
    "BarContext",
    "Clock",
    "EventEngine",
    "FillRecord",
    "MarketOrder",
    "Strategy",
    "TargetPosition",
    "TargetWeights",
    "WeightPath",
    "run_fast",
    "solve_post_trade_nav",
]

FloatArray = NDArray[np.float64]
LEDGER_COLUMNS = ("gross_ret", "cost", "borrow", "financing", "net_ret", "risk_free",
                  "excess_ret", "turnover", "nav")


# --------------------------------------------------------------------------
# clock
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Clock:
    """The shared bar clock.

    ``index`` is the bar index (dates or integers). With ``calendar`` (a
    ``qp_core.holidays`` calendar name such as ``"USD"``, or a tuple of names
    for a joint calendar) every dated bar must be a business day: a bar on a
    holiday is a data error, not a trading day, and raises.
    """

    index: pd.Index
    calendar: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        idx = pd.Index(self.index)
        object.__setattr__(self, "index", idx)
        if not idx.is_monotonic_increasing or not idx.is_unique:
            raise ValueError("bar index must be strictly increasing")
        if self.calendar is not None:
            if not isinstance(idx, pd.DatetimeIndex):
                raise ValueError("a holiday calendar needs a DatetimeIndex")
            from qp_core.holidays import is_business_day

            bad = [d.date() for d in idx if not is_business_day(d.date(), self.calendar)]
            if bad:
                raise ValueError(f"{len(bad)} bar(s) fall on non-business days of "
                                 f"{self.calendar}: first {bad[:3]}")

    def __len__(self) -> int:
        return len(self.index)


# --------------------------------------------------------------------------
# intents and records
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetWeights:
    """Trade to weights of the post-trade NAV, resolved at the fill."""

    weights: FloatArray
    tag: str = ""


@dataclass(frozen=True)
class TargetPosition:
    """Trade to share positions computed from the fill prices.

    ``resolver(fill_prices) -> target shares``. A backtest convenience: it is
    resolved into plain orders at the fill and never reaches the OMS.
    """

    resolver: Callable[[FloatArray], FloatArray]
    tag: str = ""


@dataclass(frozen=True)
class MarketOrder:
    """Buy (``quantity > 0``) or sell shares of one asset; may fill partially."""

    asset: int
    quantity: float
    tag: str = ""


Intent = TargetWeights | TargetPosition | MarketOrder


@dataclass(frozen=True)
class FillRecord:
    """One execution: what an ``oms.interface.ExecutionReport`` carries, plus
    the decision bar and the cost split."""

    order_id: int
    bar: int
    decision_bar: int
    asset: int
    quantity: float
    price: float
    cum_qty: float
    leaves_qty: float
    tag: str
    costs: dict[str, float]


@dataclass(frozen=True)
class BarContext:
    """What a strategy sees at the close of bar ``bar``."""

    bar: int
    date: Any
    prices: FloatArray
    positions: FloatArray
    cash: float
    nav: float


class Strategy(Protocol):
    def on_bar(self, ctx: BarContext) -> Iterable[Intent]: ...


@dataclass
class WeightPath:
    """Strategy that emits row ``t`` of a (T, N) target matrix at bar ``t``.

    A row of NaN means no order that bar (hold and drift).
    """

    targets: FloatArray

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        row = self.targets[ctx.bar]
        if np.isnan(row).all():
            return []
        return [TargetWeights(np.asarray(row, dtype=float))]


@dataclass
class BacktestLedger:
    """Output of both modes.

    ``ledger`` has one row per bar with :data:`LEDGER_COLUMNS` (fractions of
    the opening NAV; ``nav`` is the closing NAV). ``weights`` are the closing
    weights; event mode adds ``positions`` (shares), ``fills`` and the dollar
    split ``pnl`` (pre-fill P&L, post-fill P&L, cost and carry components).
    """

    ledger: pd.DataFrame
    weights: pd.DataFrame
    positions: pd.DataFrame | None = None
    fills: list[FillRecord] = field(default_factory=list)
    pnl: pd.DataFrame | None = None


# --------------------------------------------------------------------------
# the trade equation
# --------------------------------------------------------------------------

def solve_post_trade_nav(m: float, held: FloatArray, w: FloatArray,
                         cost: Callable[[FloatArray], float],
                         max_iter: int = 200) -> float:
    """Post-trade NAV ``N`` solving ``N = m - cost(w N - held)``.

    ``held`` is the value of the holdings at the fill and ``m`` the NAV at
    the fill, both in the same unit. The map is a contraction whenever the
    marginal cost per unit traded is below 1 / sum|w|; the iteration stops at
    machine precision and raises if it does not get there, or if the costs
    would take the post-trade NAV to zero or below.
    """
    n = m
    for _ in range(max_iter):
        n_new = m - cost(w * n - held)
        if n_new <= 0.0:
            raise RuntimeError("trade costs exceed the NAV at the fill")
        if abs(n_new - n) <= 4e-16 * max(1.0, abs(n_new)):
            return n_new
        n = n_new
    raise RuntimeError("post-trade NAV did not converge: cost per unit traded too high")


def _with_extra(cost: CostModel, extra_bp: float, bar: int) -> Callable[[FloatArray], float]:
    rate = extra_bp * 1e-4

    def c(dv: FloatArray) -> float:
        return cost(bar, dv) + rate * float(np.abs(dv).sum())

    return c


def _accruals(index: pd.Index, financing: Financing | None) -> tuple[FloatArray, FloatArray]:
    if financing is None:
        z = np.zeros(len(index))
        return z, z.copy()
    return financing.accruals(index)


def _fin(financing: Financing | None, cash_w: float, rf: float, tau: float) -> float:
    return financing.cash_return(cash_w, rf, tau) if financing is not None else 0.0


# --------------------------------------------------------------------------
# fast mode
# --------------------------------------------------------------------------

def run_fast(
    returns: pd.DataFrame,
    targets: pd.DataFrame | FloatArray,
    fill: NextBarFill | None = None,
    cost: CostModel = ZERO_COST,
    financing: Financing | None = None,
    borrow_bps_annual: float = 0.0,
    periods_per_year: float = 252.0,
) -> BacktestLedger:
    """Fast mode for a weight path.

    Parameters
    ----------
    returns : (T, N) DataFrame
        Simple returns, row ``s`` is close ``s-1`` to close ``s``. Row 0 is
        ignored (no P&L before the first close). NaN counts as 0 (the asset
        did not trade).
    targets : (T, N)
        Row ``t`` is the target decided at close ``t``; an all-NaN row means
        no order.
    fill, cost, financing, borrow_bps_annual
        See :class:`NextBarFill`, :mod:`qp_bt.costs`,
        :class:`qp_core.stats.Financing`; the borrow fee is
        ``borrow_bps_annual / periods_per_year`` per period on the short book.
    """
    fm = fill if fill is not None else NextBarFill()
    if fm.max_participation is not None:
        raise ValueError("partial fills need share volumes: use EventEngine")
    idx = returns.index
    r_all = np.nan_to_num(returns.to_numpy(dtype=float), nan=0.0)
    r_all[0] = 0.0
    tgt = np.asarray(targets, dtype=float)
    t_n, n_assets = r_all.shape
    if tgt.shape != (t_n, n_assets):
        raise ValueError(f"targets shape {tgt.shape} != returns shape {r_all.shape}")
    rf_all, tau_all = _accruals(idx, financing)
    beta = borrow_bps_annual * 1e-4 / periods_per_year
    phi = fm.fraction
    out = np.zeros((t_n, len(LEDGER_COLUMNS)))
    w_close = np.zeros((t_n, n_assets))
    h = np.zeros(n_assets)
    nav = 1.0
    for s in range(t_n):
        r = r_all[s]
        a = phi * r
        fin = _fin(financing, 1.0 - float(h.sum()), rf_all[s], tau_all[s])
        borrow = beta * float(np.sum(np.maximum(-h, 0.0) * (1.0 + a)))
        dec = s - fm.lag
        w = tgt[dec] if dec >= 0 else None
        if w is None or np.isnan(w).all():
            gross = float(h @ r)
            cost_v = turnover = 0.0
            post = h * (1.0 + r)
        else:
            if np.isnan(w).any():
                raise ValueError(f"target row {dec} is partly NaN")
            b = (1.0 + r) / (1.0 + a) - 1.0
            m = 1.0 + float(h @ a)
            held = h * (1.0 + a)
            n_post = solve_post_trade_nav(m, held, w, _with_extra(cost, fm.extra_cost_bp, dec))
            cost_v = m - n_post
            turnover = float(np.abs(w * n_post - held).sum())
            gross = m - 1.0 + n_post * float(w @ b)
            post = w * n_post * (1.0 + b)
        net = gross - cost_v - borrow + fin
        nav *= 1.0 + net
        h = post / (1.0 + net)
        w_close[s] = h
        out[s] = (gross, cost_v, borrow, fin, net, rf_all[s], net - rf_all[s], turnover, nav)
    cols = getattr(returns, "columns", None)
    return BacktestLedger(
        ledger=pd.DataFrame(out, index=idx, columns=list(LEDGER_COLUMNS)),
        weights=pd.DataFrame(w_close, index=idx, columns=cols),
    )


# --------------------------------------------------------------------------
# event mode
# --------------------------------------------------------------------------

UnitCost = Callable[[int, int, float, float], dict[str, float]]
Carry = Callable[[int, FloatArray, FloatArray], dict[str, float]]

_FILL, _DECIDE, _CLOSE = 0, 1, 2  # event priorities within one bar


@dataclass(order=True)
class _Event:
    bar: int
    priority: int
    seq: int
    kind: str = field(compare=False)
    payload: Any = field(compare=False, default=None)


class EventEngine:
    """Event-mode engine over share positions and cash.

    Each bar ``s`` is a fixed sequence of typed events taken from one queue:
    ``FILL`` events (orders decided ``lag`` bars ago fill at the fill price),
    ``DECIDE`` (the strategy sees close ``s`` and returns intents, which are
    queued as ``FILL`` events at bar ``s + lag``; with ``lag = 0`` they fill
    straight after, at close ``s``), then ``CLOSE`` (carry, financing, mark).

    Parameters
    ----------
    prices : (T, N) DataFrame of positive closes.
    fill : NextBarFill
    cost : weight-mode cost model (used by :class:`TargetWeights` and, through
        the opening NAV, by unit orders when ``unit_cost`` is not given).
    unit_cost : ``(bar, asset, dq, price) -> {component: dollars}``, optional;
        overrides ``cost`` for :class:`MarketOrder` and :class:`TargetPosition`.
    carry : ``(bar, shares_held_overnight, fill_prices) -> {component: dollars}``,
        optional; default is the borrow fee on the short book at the fill point.
    financing : qp_core.stats.Financing, optional (cash earns / pays).
    volumes : (T, N) share volumes; needed when ``fill.max_participation``.
    clock_calendar : passed to :class:`Clock`.
    """

    def __init__(self, prices: pd.DataFrame, fill: NextBarFill | None = None,
                 cost: CostModel = ZERO_COST, unit_cost: UnitCost | None = None,
                 carry: Carry | None = None, financing: Financing | None = None,
                 borrow_bps_annual: float = 0.0, periods_per_year: float = 252.0,
                 volumes: pd.DataFrame | None = None, initial_nav: float = 1.0,
                 clock_calendar: str | tuple[str, ...] | None = None) -> None:
        p = prices.to_numpy(dtype=float)
        if not np.isfinite(p).all() or (p <= 0).any():
            raise ValueError("prices must be finite and positive")
        self.fill = fill if fill is not None else NextBarFill()
        if self.fill.max_participation is not None and volumes is None:
            raise ValueError("max_participation needs volumes")
        self.prices = prices
        self.clock = Clock(prices.index, clock_calendar)
        self.p = p
        self.cost = cost
        self.unit_cost = unit_cost
        self.carry = carry
        self.financing = financing
        self.beta = borrow_bps_annual * 1e-4 / periods_per_year
        self.vol = None if volumes is None else volumes.to_numpy(dtype=float)
        self.initial_nav = float(initial_nav)

    # -- carry ---------------------------------------------------------------
    def _carry(self, s: int, q: FloatArray, pf: FloatArray) -> dict[str, float]:
        if self.carry is not None:
            return self.carry(s, q, pf)
        return {"borrow": self.beta * float(np.sum(np.maximum(-q, 0.0) * pf))}

    # -- run -----------------------------------------------------------------
    def run(self, strategy: Strategy) -> BacktestLedger:
        t_n, n = self.p.shape
        idx = self.clock.index
        rf_all, tau_all = _accruals(idx, self.financing)
        q = np.zeros(n)
        cash = self.initial_nav
        nav_prev = self.initial_nav
        queue: list[_Event] = []
        seq = itertools.count()
        order_ids = itertools.count(1)
        for s in range(t_n):
            heapq.heappush(queue, _Event(s, _DECIDE, next(seq), "decide"))
            heapq.heappush(queue, _Event(s, _CLOSE, next(seq), "close"))

        rows = np.zeros((t_n, len(LEDGER_COLUMNS)))
        pos = np.zeros((t_n, n))
        wts = np.zeros((t_n, n))
        pnl_rows: list[dict[str, float]] = []
        fills: list[FillRecord] = []
        state: dict[str, Any] = {}

        while queue:
            ev = heapq.heappop(queue)
            s = ev.bar
            p_prev = self.p[s - 1] if s > 0 else self.p[0]
            pf = p_prev + self.fill.fraction * (self.p[s] - p_prev)
            if "bar" not in state or state["bar"] != s:
                # opening of bar s: carry is set on what was held overnight
                carry = self._carry(s, q.copy(), pf) if s > 0 else {}
                cash_w = cash / nav_prev
                fin = _fin(self.financing, cash_w, rf_all[s], tau_all[s]) * nav_prev
                state = {"bar": s, "carry": carry, "fin": fin, "pf": pf,
                         "pre": float(np.sum(q * (pf - p_prev))), "cost": {}, "traded": 0.0}
            if ev.kind == "fill":
                cash, q = self._execute(ev, s, q, cash, nav_prev, state, fills, order_ids, queue,
                                        seq)
            elif ev.kind == "decide":
                ctx = BarContext(s, idx[s], self.p[s].copy(), q.copy(), cash,
                                 cash + float(q @ self.p[s]))
                lag = self.fill.lag
                for intent in strategy.on_bar(ctx):
                    if s + lag >= t_n:
                        continue  # would fill after the last bar
                    # a lag-0 fill keeps the DECIDE priority: it sorts after
                    # this event (larger seq) and before CLOSE
                    heapq.heappush(queue, _Event(s + lag, _FILL if lag else _DECIDE,
                                                 next(seq), "fill",
                                                 (next(order_ids), s, intent)))
            else:  # close
                carry_total = float(sum(state["carry"].values()))
                cash += state["fin"] - carry_total
                nav = cash + float(q @ self.p[s])
                cost_total = float(sum(state["cost"].values()))
                net = nav / nav_prev - 1.0
                c = cost_total / nav_prev
                bor = carry_total / nav_prev
                fn = state["fin"] / nav_prev
                rows[s] = (net + c + bor - fn, c, bor, fn, net, rf_all[s], net - rf_all[s],
                           state["traded"] / nav_prev, nav / self.initial_nav)
                post = float(np.sum(q * (self.p[s] - pf)))
                pnl_rows.append({"pre_fill_pnl": state["pre"], "post_fill_pnl": post, **{f"cost_{k}": v for k, v in
                                                           state["cost"].items()},
                                 **{f"carry_{k}": v for k, v in state["carry"].items()},
                                 "financing": state["fin"]})
                pos[s] = q
                wts[s] = q * self.p[s] / nav
                nav_prev = nav
        cols = self.prices.columns
        return BacktestLedger(
            ledger=pd.DataFrame(rows, index=idx, columns=list(LEDGER_COLUMNS)),
            weights=pd.DataFrame(wts, index=idx, columns=cols),
            positions=pd.DataFrame(pos, index=idx, columns=cols),
            fills=fills,
            pnl=pd.DataFrame(pnl_rows, index=idx).fillna(0.0),
        )

    # -- one fill event ------------------------------------------------------
    def _execute(self, ev: _Event, s: int, q: FloatArray, cash: float, nav_prev: float,
                 state: dict[str, Any], fills: list[FillRecord], order_ids: Any,
                 queue: list[_Event], seq: Any) -> tuple[float, FloatArray]:
        oid, dec, intent = ev.payload
        pf = state["pf"]
        extra = self.fill.extra_cost_bp * 1e-4

        def book(asset: int, dq: float, comp: dict[str, float], cum: float, leaves: float,
                 tag: str) -> None:
            for k, v in comp.items():
                state["cost"][k] = state["cost"].get(k, 0.0) + v
            state["traded"] += abs(dq) * pf[asset]
            fills.append(FillRecord(oid, s, dec, asset, dq, float(pf[asset]), cum, leaves,
                                    tag, comp))

        if isinstance(intent, TargetWeights):
            if self.fill.max_participation is not None:
                raise ValueError("partial fills apply to unit orders, not TargetWeights")
            w = np.asarray(intent.weights, dtype=float)
            held = q * pf
            m = cash + float(held.sum())
            base = _with_extra(self.cost, 0.0, dec)

            def c_dollars(dv: FloatArray) -> float:
                return nav_prev * base(dv / nav_prev) + extra * float(np.abs(dv).sum())

            n_post = solve_post_trade_nav(m, held, w, c_dollars)
            q_new = w * n_post / pf
            dq = q_new - q
            total = m - n_post
            lin = extra * float(np.abs(dq * pf).sum())
            comp = {"model": total - lin, "extra": lin} if extra else {"model": total}
            for k, v in comp.items():
                state["cost"][k] = state["cost"].get(k, 0.0) + v
            for i in np.flatnonzero(dq):
                state["traded"] += abs(dq[i]) * pf[i]
                fills.append(FillRecord(oid, s, dec, int(i), float(dq[i]), float(pf[i]),
                                        abs(float(dq[i])), 0.0, intent.tag, {}))
            # cash after the trade is N minus the new book's value, exactly
            return n_post - float(q_new @ pf), q_new

        if isinstance(intent, TargetPosition):
            target = np.asarray(intent.resolver(pf.copy()), dtype=float)
            orders = [(i, float(target[i] - q[i])) for i in range(len(q))]
        else:
            orders = [(intent.asset, float(intent.quantity))]
        q = q.copy()
        for asset, dq in orders:
            want = dq
            if self.fill.max_participation is not None and self.vol is not None:
                cap = self.fill.max_participation * float(self.vol[s, asset])
                dq = float(np.sign(dq) * min(abs(dq), cap))
            if self.unit_cost is not None:
                comp = self.unit_cost(s, asset, dq, float(pf[asset]))
            else:
                dv = np.zeros(len(q))
                dv[asset] = dq * pf[asset] / nav_prev
                comp = {"model": nav_prev * self.cost(dec, dv)}
            if extra:
                comp = {**comp, "extra": extra * abs(dq) * float(pf[asset])}
            leaves = abs(want - dq)
            if dq != 0.0 or self.unit_cost is not None:
                book(asset, dq, comp, abs(dq), leaves, getattr(intent, "tag", ""))
            q[asset] += dq
            cash -= dq * float(pf[asset]) + float(sum(comp.values()))
            if leaves > 0.0 and s + 1 < len(self.p):
                rest = MarketOrder(asset, want - dq, getattr(intent, "tag", ""))
                heapq.heappush(queue, _Event(s + 1, _FILL, next(seq), "fill", (oid, dec, rest)))
        return cash, q
