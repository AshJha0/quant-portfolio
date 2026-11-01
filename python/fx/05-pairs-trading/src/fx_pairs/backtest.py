"""Daily backtest for pairs of currency pairs, with carry and pip costs.

Execution convention (shared by every strategy project in the repository)
-------------------------------------------------------------------------
Signals use data up to the close of day t. Orders fill at a modelled t+1
price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
variant fills at p_t and pays an auction cost. For daily FX the t+1 price is
the next daily close. See :class:`ExecutionConfig`.

Accounting (all P&L as a fraction of base notional; multiply by ``notional``).
``pos_t`` is the spread position decided at the close of ``t``; leg notionals
are ``n1_t = pos_t`` (long pair 1) and ``n2_t = pos_t * beta_t`` (short pair 2).

Mode ``"next"`` (default). The order for ``pos_{t-1}`` fills during period
``t`` at ``q_t = p_{t-1} + f (p_t - p_{t-1})``. Booked at ``t``:

* spot: ``n_{t-2} (log q_t - log p_{t-1}) + n_{t-1} (log p_t - log q_t)``
  per leg (pair 2 with a minus sign). The pieces sum to the log return.
* carry: ``(f n_{t-2} + (1 - f) n_{t-1}) * accr_t``, accruals from rates at
  ``t-1`` and the calendar-day gap (ACT/365F).
* cost: ``|n_{t-1} - n_{t-2}| * hs`` with half-spread fraction
  ``hs = 0.5 * pip_spread * pip_size / q_t``.

Mode ``"close_auction"``. The order for ``pos_t`` fills at ``p_t``. Spot is
``n_{t-1} dlog p_t``, carry ``n_{t-1} accr_t``, and the cost at ``t`` is
``|n_t - n_{t-1}| * (hs_t + auction_cost_bp * 1e-4)``.

Identity (unit tested): ``total = spot + carry + cost`` elementwise, with
``cost <= 0``. No lookahead: P&L booked at ``t`` uses prices up to ``t`` and
positions decided before ``t``. A detector test perturbs prices after day
``k`` and asserts P&L through ``k`` is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._validation import finite_series, require_finite

from . import carry as carry_mod
from .cointegration import engle_granger
from .signals import Trade, generate_positions, zscore
from .universe import pip_size as _pip_size

__all__ = [
    "ExecutionConfig",
    "BacktestResult",
    "run_backtest",
    "WalkForwardWindow",
    "WalkForwardResult",
    "walk_forward_backtest",
]


@dataclass(frozen=True)
class ExecutionConfig:
    """When and at what price a decision taken at the close of ``t`` fills.

    Parameters
    ----------
    mode : str
        ``"next"``: fill at the modelled t+1 price ``p_t + f (p_{t+1} - p_t)``.
        ``"close_auction"``: fill at ``p_t`` and pay ``auction_cost_bp`` per
        unit of traded notional on top of the half spread.
    fill_fraction : float
        ``f`` in (0, 1]; 1 is the next close. Used by ``"next"`` only.
    auction_cost_bp : float
        Extra cost of a close-auction fill, basis points of traded notional.
    """

    mode: str = "next"
    fill_fraction: float = 0.5
    auction_cost_bp: float = 10.0

    def __post_init__(self) -> None:
        if self.mode not in ("next", "close_auction"):
            raise ValueError(f"mode must be 'next' or 'close_auction', got {self.mode!r}")
        require_finite(fill_fraction=self.fill_fraction,
                       auction_cost_bp=self.auction_cost_bp)
        if not 0.0 < self.fill_fraction <= 1.0:
            raise ValueError(f"fill_fraction must be in (0, 1], got {self.fill_fraction}")
        if self.auction_cost_bp < 0:
            raise ValueError("auction_cost_bp must be non-negative")


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
    execution: ExecutionConfig | None = None,
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

    lp1 = np.log(finite_series(p1.to_numpy(dtype=float), "p1", positive=True))
    lp2 = np.log(finite_series(p2.to_numpy(dtype=float), "p2", positive=True))

    ex = execution if execution is not None else ExecutionConfig()
    if not isinstance(ex, ExecutionConfig):
        raise ValueError("execution must be an ExecutionConfig")
    px1 = p1.to_numpy(dtype=float)
    px2 = p2.to_numpy(dtype=float)

    n1 = pos
    n2 = pos * beta_s
    accr1 = accr2 = None
    if rates is not None:
        for key in ("rb1", "rq1", "rb2", "rq2"):
            if key not in rates:
                raise ValueError(f"rates dict missing key {key!r}")
            # Named by dict key so the error points at the offending quote.
            require_finite(**{f"rates[{key!r}]": rates[key]})
        accr1 = carry_mod.carry_accrual(rates["rb1"], rates["rq1"], index,
                                        basis=basis, method=carry_method)
        accr2 = carry_mod.carry_accrual(rates["rb2"], rates["rq2"], index,
                                        basis=basis, method=carry_method)
    ps1 = 0.5 * pip_spread_1 * _pip_size(pair1)
    ps2 = 0.5 * pip_spread_2 * _pip_size(pair2)

    spot = np.zeros(n)
    car = np.zeros(n)
    cost = np.zeros(n)
    if ex.mode == "close_auction":
        spot[1:] = n1[:-1] * np.diff(lp1) - n2[:-1] * np.diff(lp2)
        if accr1 is not None:
            car[1:] = n1[:-1] * accr1[1:] - n2[:-1] * accr2[1:]
        dn1 = np.abs(np.diff(np.concatenate([[0.0], n1])))
        dn2 = np.abs(np.diff(np.concatenate([[0.0], n2])))
        auc = ex.auction_cost_bp * 1e-4
        cost = -(dn1 * (ps1 / px1 + auc) + dn2 * (ps2 / px2 + auc))
    else:
        f = ex.fill_fraction
        # held before / after the fill inside period t (t = 1..n-1)
        old1 = np.concatenate([[0.0], n1[:-2]])
        old2 = np.concatenate([[0.0], n2[:-2]])
        new1, new2 = n1[:-1], n2[:-1]
        q1 = px1[:-1] + f * np.diff(px1)
        q2 = px2[:-1] + f * np.diff(px2)
        lq1, lq2 = np.log(q1), np.log(q2)
        spot[1:] = (old1 * (lq1 - lp1[:-1]) + new1 * (lp1[1:] - lq1)
                    - old2 * (lq2 - lp2[:-1]) - new2 * (lp2[1:] - lq2))
        if accr1 is not None:
            car[1:] = ((f * old1 + (1 - f) * new1) * accr1[1:]
                       - (f * old2 + (1 - f) * new2) * accr2[1:])
        cost[1:] = -(np.abs(new1 - old1) * ps1 / q1 + np.abs(new2 - old2) * ps2 / q2)

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
