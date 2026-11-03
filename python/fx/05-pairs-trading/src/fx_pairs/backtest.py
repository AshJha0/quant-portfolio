"""Daily backtest for pairs of currency pairs, with carry and pip costs.

Execution convention (shared by every strategy project in the repository)
-------------------------------------------------------------------------
Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost. For daily FX the t+1 price is
the next daily close. See :class:`ExecutionConfig`.

Accounting, on the shared ``qp_bt`` event engine (Round 12, W24b)
------------------------------------------------------------------
``pos_t`` is the spread position decided at the close of ``t``. Its legs are
quote-currency (USD) exposures ``pos_t * notional`` in pair 1 and
``-pos_t * beta_t * notional`` in pair 2, sized at the fill price: the order
buys ``e / q`` units of base, where ``q`` is the fill price. The units are
then held, and their USD value moves with the price, until the position
changes. All P&L is in units of account of the quote currency.

Mode ``"next"`` (default). An order decided at ``t-1`` fills in period
``t`` at ``q_t = p_{t-1} + f (p_t - p_{t-1})``. Booked at ``t``, per leg
with ``u`` units of base held before the fill and ``u'`` after:

* spot: ``u (q_t - p_{t-1}) + u' (p_t - q_t)``;
* carry: ``(f u + (1 - f) u') p_{t-1} accr_t``, accruals from rates at
  ``t-1`` and the calendar-day gap (ACT/365F), time-weighted across the
  fill as before;
* cost: ``-|u' - u| * hs`` with half-spread ``hs = 0.5 * pip_spread *
  pip_size`` in price units.

Mode ``"close_auction"``. The order for ``pos_t`` fills at ``p_t``, booked
at ``t``: spot ``u (p_t - p_{t-1})``, carry ``u p_{t-1} accr_t`` and cost
``-|u' - u| (hs + auction_cost_bp * 1e-4 * p_t)``.

Identity (unit tested): ``total = spot + carry + cost + financing``
elementwise, with ``cost <= 0``. No lookahead: P&L booked at ``t`` uses
prices up to ``t`` and positions decided before ``t``. A detector test
perturbs prices after day ``k`` and asserts P&L through ``k`` is unchanged.

Migration (W24b). Before, the P&L was ``n * notional * dlog p`` with the
exposure reset to ``n * notional`` every day for free: log returns on a
constant USD exposure. Now the exposure is bought once at the fill and
drifts; spot P&L is the USD change in value of the units held. Before/after
numbers are in the handback; hand fixtures are in ``tests/test_backtest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from qp_bt import BarContext, EventEngine, NextBarFill, TargetPosition
from qp_core.execution import ExecutionConfig
from qp_core.stats import Financing, warn_no_financing

from . import carry as carry_mod
from ._validation import finite_series, require_finite
from .cointegration import engle_granger
from .signals import Trade, generate_positions
from .universe import pip_size as _pip_size

__all__ = [
    "BacktestResult",
    "ExecutionConfig",
    "WalkForwardResult",
    "WalkForwardWindow",
    "run_backtest",
    "walk_forward_backtest",
]



@dataclass
class BacktestResult:
    """Backtest output with full P&L decomposition.

    All P&L series are in units of account (base notional * fractional
    returns).  ``total_pnl = spot_pnl + carry_pnl + cost_pnl`` exactly
    (``cost_pnl <= 0``).
    """

    positions: pd.Series
    spot_pnl: pd.Series
    carry_pnl: pd.Series
    cost_pnl: pd.Series
    total_pnl: pd.Series
    equity: pd.Series
    beta: pd.Series
    notional: float
    trades: list[Trade] = field(default_factory=list)
    leg1_pnl: pd.Series | None = None
    leg2_pnl: pd.Series | None = None
    financing_pnl: pd.Series | None = None
    risk_free_pnl: pd.Series | None = None
    units: pd.DataFrame | None = None
    """Base-currency units held per leg at each close (``leg1``, ``leg2``;
    leg 2 is negative when the spread is long). The P&L is earned on these."""

    @property
    def excess_pnl(self) -> pd.Series:
        """P&L over the risk-free return on the posted capital: spot + carry +
        cost. The FX legs are self-financing, so this is the excess P&L
        whether or not financing was given; use it for Sharpe (W6d)."""
        return (self.spot_pnl + self.carry_pnl + self.cost_pnl).rename("excess_pnl")

    def decomposition(self) -> dict[str, float]:
        """Cumulative P&L split: spot vs carry vs costs (sums of the ledgers)."""
        return {
            "spot": float(self.spot_pnl.sum()),
            "carry": float(self.carry_pnl.sum()),
            "costs": float(self.cost_pnl.sum()),
            "total": float(self.total_pnl.sum()),
        }


def _as_series(x: float | pd.Series, index: pd.Index, name: str) -> pd.Series:
    if np.isscalar(x):
        return pd.Series(float(x), index=index, name=name)
    if not x.index.equals(index):
        raise ValueError(f"{name} index must match price index")
    return x.astype(float)


def run_backtest(
    p1: pd.Series,
    p2: pd.Series,
    positions: pd.Series | np.ndarray,
    beta: float | pd.Series,
    pair1: str = "XXXUSD",
    pair2: str = "XXXUSD",
    pip_spread_1: float = 1.0,
    pip_spread_2: float = 1.0,
    rates: dict[str, float | pd.Series] | None = None,
    notional: float = 1.0,
    basis: float = 365.0,
    carry_method: str = "swap",
    trades: list[Trade] | None = None,
    execution: ExecutionConfig | None = None,
    financing: Financing | None = None,
    capital: float | None = None,
) -> BacktestResult:
    """Run the daily pairs backtest (see module docstring for accounting).

    Parameters
    ----------
    p1, p2 : pandas.Series
        Spot prices of the two currency pairs on a common date index.
    positions : array-like
        Spread position (float, spread units) decided at the close of t.
    beta : float or pandas.Series
        Hedge ratio (pair-2 units per pair-1 unit).
    pair1, pair2 : str
        6-letter pair codes: used only for pip size (0.01 for ``...JPY``).
    pip_spread_1, pip_spread_2 : float
        Full quoted bid-ask spread in pips; half is paid per side.
    rates : dict, optional
        ``{"rb1": .., "rq1": .., "rb2": .., "rq2": ..}``: annualised deposit
        rates (scalars or Series) for base/quote of each pair.  ``None`` =>
        spot-only backtest (zero carry).
    notional : float
        Base notional per spread unit, units of account.
    basis : float
        Carry day-count basis (365 = ACT/365F).
    carry_method : str
        ``"swap"`` (exact swap points) or ``"linear"``.
    trades : list, optional
        Trade list from the signal generator, carried through for metrics.
    execution : ExecutionConfig, optional
        Fill convention; default ``ExecutionConfig()`` (next-period modelled
        fill, ``f = 0.5``).
    financing : qp_core.stats.Financing, optional
        Rate earned on the posted ``capital`` (W6d). The FX legs are
        self-financing (carry is in ``carry_pnl``), so financing is the
        interest on collateral: ``financing_pnl = capital * (rf - lend
        spread) tau``, added to ``total_pnl``; ``risk_free_pnl = capital *
        rf``. ``None`` means the capital earns 0 and emits
        ``FinancingWarning``.
    capital : float, optional
        Collateral in units of account; default ``notional``.

    Numeraire: every P&L series is in units of account of the quote
    currency of the pairs (USD for ``XXXUSD``), per ``notional`` of base.
    ``leg1_pnl`` and ``leg2_pnl`` split spot + carry + cost by leg. Leg 2
    is held short, and its series already carries that sign, so
    ``leg1_pnl + leg2_pnl = total_pnl - financing_pnl``.

    Returns
    -------
    BacktestResult
    """
    if not p1.index.equals(p2.index):
        raise ValueError("p1 and p2 must share the same index")
    n = len(p1)
    if n < 2:
        raise ValueError("need at least two observations")
    pos = np.asarray(positions, dtype=float)
    if len(pos) != n:
        raise ValueError(f"positions length {len(pos)} != prices length {n}")
    if not np.isfinite(pos).all():
        raise ValueError(
            "positions contain NaN or infinite values (treat warmup as 0)"
        )
    require_finite(pip_spread_1=pip_spread_1, pip_spread_2=pip_spread_2,
                   notional=notional, basis=basis)
    if pip_spread_1 < 0 or pip_spread_2 < 0:
        raise ValueError("pip spreads must be non-negative")
    if notional <= 0:
        raise ValueError("notional must be positive")
    if basis <= 0:
        raise ValueError("basis must be positive")
    index = p1.index
    beta_s = _as_series(beta, index, "beta").to_numpy()
    require_finite(beta=beta_s)

    finite_series(p1.to_numpy(dtype=float), "p1", positive=True)
    finite_series(p2.to_numpy(dtype=float), "p2", positive=True)

    ex = execution if execution is not None else ExecutionConfig()
    if not isinstance(ex, ExecutionConfig):
        raise ValueError("execution must be an ExecutionConfig")
    px1 = p1.to_numpy(dtype=float)
    px2 = p2.to_numpy(dtype=float)

    e1 = pos * notional            # USD exposure, pair 1 (long the spread)
    e2 = -pos * beta_s * notional  # USD exposure, pair 2 (short beta per unit)
    accr = np.zeros((n, 2))
    if rates is not None:
        for key in ("rb1", "rq1", "rb2", "rq2"):
            if key not in rates:
                raise ValueError(f"rates dict missing key {key!r}")
            # Named by dict key so the error points at the offending quote.
            require_finite(**{f"rates[{key!r}]": rates[key]})
        accr[:, 0] = carry_mod.carry_accrual(rates["rb1"], rates["rq1"], index,
                                             basis=basis, method=carry_method)
        accr[:, 1] = carry_mod.carry_accrual(rates["rb2"], rates["rq2"], index,
                                             basis=basis, method=carry_method)
    hs = np.array([0.5 * pip_spread_1 * _pip_size(pair1), 0.5 * pip_spread_2 * _pip_size(pair2)])
    px = np.column_stack([px1, px2])
    if ex.mode == "close_auction":
        fill = NextBarFill.from_execution(ex, auction_booked="decision")
        f = 0.0
    else:
        fill = NextBarFill.from_execution(ex)
        f = fill.fraction
    lag = fill.lag
    expo = np.column_stack([e1, e2])
    changed = np.r_[True, np.any(np.diff(expo, axis=0) != 0.0, axis=1)]
    changed &= ~((np.arange(n) == 0) & ~np.any(expo[0] != 0.0))

    def target_units(t: int, pf: np.ndarray) -> np.ndarray:
        return np.asarray(expo[t] / pf, dtype=float)

    def unit_cost(s: int, asset: int, dq: float, price: float) -> dict[str, float]:
        return {f"leg{asset + 1}": abs(dq) * float(hs[asset])}

    def leg_carry(s: int, q: np.ndarray, pf: np.ndarray) -> dict[str, float]:
        held = q
        dec = s - lag
        if f > 0.0 and dec >= 0 and changed[dec]:
            held = f * q + (1.0 - f) * target_units(dec, pf)
        c = held * px[s - 1] * accr[s]
        return {"leg1": -float(c[0]), "leg2": -float(c[1])}

    class _Path:
        def on_bar(self, ctx: BarContext) -> list[TargetPosition]:
            t = ctx.bar
            if not changed[t]:
                return []
            def resolve(pf: np.ndarray) -> np.ndarray:
                return target_units(t, pf)

            return [TargetPosition(resolve)]

    # NAV only has to stay positive for the engine's wipe-out rule; P&L is
    # read from its per-leg split, not from NAV.
    nav0 = 1.0 + 1e6 * (1.0 + float(np.abs(expo).sum(axis=1).max(initial=0.0)))
    run = EventEngine(pd.DataFrame(px, index=pd.RangeIndex(n)), fill, unit_cost=unit_cost,
                      carry=leg_carry, initial_nav=nav0).run(_Path())
    assert run.positions is not None and run.pnl is not None
    units = run.positions.to_numpy(dtype=float)
    before = np.vstack([np.zeros((1, 2)), units[:-1]])
    after = units
    p_prev = np.vstack([px[:1], px[:-1]])
    q_fill = p_prev + fill.fraction * (px - p_prev)
    leg_spot = before * (q_fill - p_prev) + after * (px - q_fill)
    leg_spot[0] = 0.0
    leg_cost = np.zeros((n, 2))
    for rec in run.fills:
        leg_cost[rec.bar, rec.asset] -= float(sum(rec.costs.values()))
    leg_car = np.zeros((n, 2))
    for j in (0, 1):
        col = f"carry_leg{j + 1}"
        if col in run.pnl:
            leg_car[:, j] = -run.pnl[col].to_numpy(dtype=float)
    spot = leg_spot.sum(axis=1)
    car = leg_car.sum(axis=1)
    cost = leg_cost.sum(axis=1)
    leg1 = leg_spot[:, 0] + leg_car[:, 0] + leg_cost[:, 0]
    leg2 = leg_spot[:, 1] + leg_car[:, 1] + leg_cost[:, 1]

    spot_pnl = pd.Series(spot, index=index, name="spot_pnl")
    carry_pnl = pd.Series(car, index=index, name="carry_pnl")
    cost_pnl = pd.Series(cost, index=index, name="cost_pnl")
    cap = notional if capital is None else float(capital)
    if financing is None:
        warn_no_financing("fx_pairs.run_backtest")
        rf = np.zeros(n)
        fin = np.zeros(n)
    else:
        rf, tau = financing.accruals(index)
        fin = np.array([financing.cash_return(1.0, rf[t], tau[t]) for t in range(n)])
    fin_pnl = pd.Series(fin * cap, index=index, name="financing_pnl")
    total = spot_pnl + carry_pnl + cost_pnl + fin_pnl
    total.name = "total_pnl"
    return BacktestResult(
        positions=pd.Series(pos, index=index, name="position"),
        spot_pnl=spot_pnl, carry_pnl=carry_pnl, cost_pnl=cost_pnl,
        total_pnl=total, equity=total.cumsum().rename("equity"),
        beta=pd.Series(beta_s, index=index, name="beta"),
        notional=notional, trades=list(trades) if trades else [],
        leg1_pnl=pd.Series(leg1, index=index, name="leg1_pnl"),
        leg2_pnl=pd.Series(leg2, index=index, name="leg2_pnl"),
        financing_pnl=fin_pnl,
        risk_free_pnl=pd.Series(rf * cap, index=index, name="risk_free_pnl"),
        units=pd.DataFrame(units, index=index, columns=["leg1", "leg2"]),
    )


@dataclass
class WalkForwardWindow:
    """One formation/trading block of the walk-forward.

    Index positions are half-open: formation ``[f0, f1)``, trading
    ``[t0, t1)``, with ``f1 == t0`` (formation strictly precedes trading:
    no overlap, no lookahead).
    """

    f0: int
    f1: int
    t0: int
    t1: int
    alpha: float
    beta: float
    mu: float
    sigma: float
    eg_stat: float
    cointegrated: bool
    degenerate: bool
    traded: bool


@dataclass
class WalkForwardResult:
    """Stitched walk-forward output."""

    result: BacktestResult
    windows: list[WalkForwardWindow]
    positions: pd.Series


def walk_forward_backtest(
    p1: pd.Series,
    p2: pd.Series,
    formation: int = 252,
    trading: int = 63,
    entry: float = 2.0,
    exit_: float = 0.5,
    stop: float | None = 4.0,
    max_holding: int | None = None,
    require_coint: bool = True,
    coint_level: str = "10%",
    **backtest_kwargs,
) -> WalkForwardResult:
    """Rolling formation/trading walk-forward with frozen per-window parameters.

    For each block: fit Engle-Granger (``log p1`` on ``log p2``) on the
    formation window; freeze ``(alpha, beta, mu, sigma)`` from the formation
    residuals; in the following trading window compute the spread and z-score
    with those frozen parameters, run the signal state machine, and force flat
    at the window's last bar.  Windows tile the sample with no overlap between
    a window's formation and its trading data.

    Parameters
    ----------
    formation, trading : int
        Window lengths in bars.
    require_coint : bool
        Skip (stay flat in) trading windows whose formation Engle-Granger
        statistic does not reject at ``coint_level``, or is degenerate.
    **backtest_kwargs
        Passed through to :func:`run_backtest` (costs, rates, notional...).

    Returns
    -------
    WalkForwardResult
    """
    if not p1.index.equals(p2.index):
        raise ValueError("p1 and p2 must share the same index")
    n = len(p1)
    if formation < 60 or trading < 5:
        raise ValueError("formation must be >= 60 and trading >= 5 bars")
    if n < formation + trading:
        raise ValueError("sample too short for one walk-forward block")

    lp1 = np.log(p1.to_numpy(dtype=float))
    lp2 = np.log(p2.to_numpy(dtype=float))
    pos = np.zeros(n)
    beta_series = np.zeros(n)
    windows: list[WalkForwardWindow] = []
    all_trades: list[Trade] = []

    t0 = formation
    while t0 + 1 < n:
        t1 = min(t0 + trading, n)
        f0, f1 = t0 - formation, t0
        eg = engle_granger(lp1[f0:f1], lp2[f0:f1])
        mu = float(np.mean(eg.resid))
        sigma = float(np.std(eg.resid, ddof=1))
        traded = (not eg.degenerate) and sigma > 0 and (
            not require_coint or eg.stat < eg.crit_values[coint_level]
        )
        if traded:
            sp = lp1[t0:t1] - eg.alpha - eg.beta * lp2[t0:t1]
            z = (sp - mu) / sigma
            w_pos, w_trades = generate_positions(
                pd.Series(z), entry=entry, exit_=exit_, stop=stop,
                max_holding=max_holding,
            )
            w_pos[-1] = 0.0  # force flat at window end
            pos[t0:t1] = w_pos
            beta_series[t0:t1] = eg.beta
            for tr in w_trades:
                all_trades.append(
                    Trade(entry=tr.entry + t0, exit=tr.exit + t0,
                          side=tr.side, exit_reason=tr.exit_reason)
                )
        windows.append(WalkForwardWindow(
            f0=f0, f1=f1, t0=t0, t1=t1, alpha=eg.alpha, beta=eg.beta,
            mu=mu, sigma=sigma, eg_stat=eg.stat,
            cointegrated=eg.cointegrated, degenerate=eg.degenerate,
            traded=traded,
        ))
        t0 = t1

    beta_s = pd.Series(beta_series, index=p1.index, name="beta")
    result = run_backtest(p1, p2, pos, beta_s, trades=all_trades, **backtest_kwargs)
    return WalkForwardResult(
        result=result, windows=windows,
        positions=pd.Series(pos, index=p1.index, name="position"),
    )
