"""Daily event-driven backtest for pairs of currency pairs, with carry and pip costs.

Accounting (all P&L as a fraction of base notional; multiply by ``notional``):

* Leg notionals at close ``t``: ``n1_t = pos_t`` (units long pair 1) and
  ``n2_t = pos_t * beta_t`` (units short pair 2).  ``pos_t`` may be any float
  (vol-targeted sizing); ``beta`` may be a scalar or a per-date Series
  (walk-forward / RLS hedge).
* Spot P&L booked at ``t``: ``n1_{t-1} * dlog p1_t - n2_{t-1} * dlog p2_t``
  (log-return P&L — exact for the log-spread the signals trade).
* Carry booked at ``t``: ``n1_{t-1} * accr1_t - n2_{t-1} * accr2_t`` where the
  accruals use rates at ``t-1`` and the actual calendar-day gap
  (:func:`fx_pairs.carry.carry_accrual`, swap-point form by default).
* Costs booked at ``t`` (trade executed at close ``t``):
  ``|n1_t - n1_{t-1}| * hs1_t + |n2_t - n2_{t-1}| * hs2_t`` where the
  half-spread fraction is ``hs_t = 0.5 * pip_spread * pip_size / price_t``
  (pay half the quoted bid-ask spread per side, per unit of notional traded).
* Identity (unit tested): ``total = spot + carry + cost`` elementwise, with
  ``cost <= 0``; and running with rates equals running without rates plus the
  independent carry ledger.

No lookahead: ``P&L_t`` depends only on data up to ``t`` — positions are
functions of information through their own bar, returns are applied one bar
later, carry uses lagged rates.  The test suite includes a detector that
perturbs prices after day ``k`` and asserts P&L through ``k`` is bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import carry as carry_mod
from .cointegration import engle_granger
from .signals import Trade, generate_positions, zscore
from .universe import pip_size as _pip_size

__all__ = [
    "BacktestResult",
    "run_backtest",
    "WalkForwardWindow",
    "WalkForwardResult",
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
) -> BacktestResult:
    """Run the daily pairs backtest (see module docstring for accounting).

    Parameters
    ----------
    p1, p2 : pandas.Series
        Spot prices of the two currency pairs on a common date index.
    positions : array-like
        Spread position (float, spread units) held from close t to close t+1.
    beta : float or pandas.Series
        Hedge ratio (pair-2 units per pair-1 unit).
    pair1, pair2 : str
        6-letter pair codes — used only for pip size (0.01 for ``...JPY``).
    pip_spread_1, pip_spread_2 : float
        Full quoted bid-ask spread in pips; half is paid per side.
    rates : dict, optional
        ``{"rb1": .., "rq1": .., "rb2": .., "rq2": ..}`` — annualised deposit
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
    if np.isnan(pos).any():
        raise ValueError("positions contain NaNs (treat warmup as 0)")
    if pip_spread_1 < 0 or pip_spread_2 < 0:
        raise ValueError("pip spreads must be non-negative")
    if notional <= 0:
        raise ValueError("notional must be positive")
    index = p1.index
    beta_s = _as_series(beta, index, "beta").to_numpy()

    lp1 = np.log(p1.to_numpy(dtype=float))
    lp2 = np.log(p2.to_numpy(dtype=float))

    n1 = pos
    n2 = pos * beta_s

    spot = np.zeros(n)
    spot[1:] = n1[:-1] * np.diff(lp1) - n2[:-1] * np.diff(lp2)

    car = np.zeros(n)
    if rates is not None:
        for key in ("rb1", "rq1", "rb2", "rq2"):
            if key not in rates:
                raise ValueError(f"rates dict missing key {key!r}")
        accr1 = carry_mod.carry_accrual(rates["rb1"], rates["rq1"], index,
                                        basis=basis, method=carry_method)
        accr2 = carry_mod.carry_accrual(rates["rb2"], rates["rq2"], index,
                                        basis=basis, method=carry_method)
        car[1:] = n1[:-1] * accr1[1:] - n2[:-1] * accr2[1:]

    hs1 = 0.5 * pip_spread_1 * _pip_size(pair1) / p1.to_numpy(dtype=float)
    hs2 = 0.5 * pip_spread_2 * _pip_size(pair2) / p2.to_numpy(dtype=float)
    dn1 = np.abs(np.diff(np.concatenate([[0.0], n1])))
    dn2 = np.abs(np.diff(np.concatenate([[0.0], n2])))
    cost = -(dn1 * hs1 + dn2 * hs2)

    spot_pnl = pd.Series(spot * notional, index=index, name="spot_pnl")
    carry_pnl = pd.Series(car * notional, index=index, name="carry_pnl")
    cost_pnl = pd.Series(cost * notional, index=index, name="cost_pnl")
    total = spot_pnl + carry_pnl + cost_pnl
    total.name = "total_pnl"
    return BacktestResult(
        positions=pd.Series(pos, index=index, name="position"),
        spot_pnl=spot_pnl, carry_pnl=carry_pnl, cost_pnl=cost_pnl,
        total_pnl=total, equity=total.cumsum().rename("equity"),
        beta=pd.Series(beta_s, index=index, name="beta"),
        notional=notional, trades=list(trades) if trades else [],
    )


@dataclass
class WalkForwardWindow:
    """One formation/trading block of the walk-forward.

    Index positions are half-open: formation ``[f0, f1)``, trading
    ``[t0, t1)``, with ``f1 == t0`` (formation strictly precedes trading —
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
