"""Walk-forward backtesting for currency portfolios, on the shared ``qp_bt`` core.

Design rules (all tested):

* **No lookahead**: the weight function receives history strictly up to and
  including the rebalance date *t*; the order fills in period *t+1* under
  ``execution`` (default: next bar, ``f = 0.5``, the repository convention
  in :class:`qp_core.execution.ExecutionConfig`).
* **One core**: the period arithmetic is :func:`qp_bt.run_fast`: the book is
  traded to target weights of the post-trade NAV, costs come out of the
  trade, and between rebalances the book drifts with prices (no free daily
  reset to target). Spot log returns are turned into simple returns,
  ``R = exp(x) - 1``, before they reach the core.
* **Carry in the P&L**: the ledger splits each day into a spot leg (the
  core's ``gross_ret``) and a carry leg (``asset_carry``: the accrual of each
  currency, earned on the book held overnight). Net = spot + carry - cost.
* **Transaction costs in pips**: :func:`pips_to_bps` converts
  ``pips * pip_size / spot`` into bps of notional; the core charges
  ``cost_bps * 1e-4`` per unit of traded notional (:class:`qp_bt.HalfSpreadCost`).
* **Base-currency reporting**: helpers convert USD-base log total returns to
  any base (the GBP option for a London desk). For LOG returns the change of
  base adds the SAME conversion term to every asset, so a dollar-neutral
  portfolio's linearly aggregated log-return series is EXACTLY invariant to
  the base currency (the common term multiplies sum(w) = 0); a net-long book
  shifts by the base currency's own total return. Both identities are
  unit-tested.

Migration (Round 12, W24b). Before, the book earned the whole day after the
decision close (a free fill at the decision close), weights were reset to
target every day for free, and the day's P&L was the weighted sum of log
returns. Each is listed with before/after numbers in the handback and
checked by a hand fixture in ``tests/test_backtest.py``. The old fill price
is ``execution=ExecutionConfig("close_auction", auction_cost_bp=0.0)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from qp_bt import HalfSpreadCost, NextBarFill, run_fast
from qp_core.execution import ExecutionConfig

from .returns_est import DT


def pips_to_bps(pips: float, spot: float, pip_size: float = 1e-4) -> float:
    """Convert a cost quoted in pips to basis points of notional.

    Parameters
    ----------
    pips : float
        Cost in pips (e.g. half the quoted spread), >= 0.
    spot : float
        Spot level in the pair's own quoting convention, > 0.
    pip_size : float
        Price value of one pip: 1e-4 for most pairs, 1e-2 for JPY pairs.

    Returns
    -------
    float
        Cost in bps: ``pips * pip_size / spot * 1e4``.
    """
    if pips < 0:
        raise ValueError(f"pips must be >= 0, got {pips}")
    if spot <= 0 or pip_size <= 0:
        raise ValueError("spot and pip_size must be > 0")
    return pips * pip_size / spot * 1e4


@dataclass
class BacktestResult:
    """Walk-forward output.

    Attributes
    ----------
    ledger : pd.DataFrame
        Daily rows, fractions of the previous close's NAV: ``spot_pnl``,
        ``carry_pnl``, ``cost``, ``net`` (= spot + carry - cost, a simple
        return) and ``turnover`` (traded notional over that NAV).
    weights : pd.DataFrame
        Weights held at each close (after the day's fill and drift).
    """

    ledger: pd.DataFrame
    weights: pd.DataFrame


def run_backtest(
    spot_returns: pd.DataFrame,
    carry_returns: pd.DataFrame,
    weight_func: Callable[[pd.DataFrame], pd.Series],
    est_window: int = 252,
    rebalance_every: int = 21,
    cost_bps: float | pd.Series = 1.0,
    execution: ExecutionConfig | None = None,
) -> BacktestResult:
    """Walk-forward backtest with rebalancing, pip costs and carry accrual.

    Parameters
    ----------
    spot_returns, carry_returns : pd.DataFrame
        Daily spot log returns and carry accrual per asset (same shape;
        their sum is the total log return). Obtain them from
        :func:`fx_port.returns_est.total_log_returns`.
    weight_func : callable
        ``weight_func(history)`` -> weights Series, where ``history`` is the
        TOTAL return panel up to and including the rebalance date. Called
        first on row ``est_window - 1`` and then every ``rebalance_every``
        days.
    est_window : int
        Minimum history before the first allocation (>= 2).
    rebalance_every : int
        Days between rebalances (>= 1).
    cost_bps : float or pd.Series
        One-way cost per unit of traded notional, in bps, scalar or per asset.
    execution : ExecutionConfig, optional
        Fill convention; default next bar, ``f = 0.5``. A close-auction fill
        is booked in the next day's row.

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError
        On mismatched panels, too-short samples or invalid parameters.
    """
    if not spot_returns.index.equals(carry_returns.index) or list(
        spot_returns.columns
    ) != list(carry_returns.columns):
        raise ValueError("spot_returns and carry_returns must be aligned")
    n_days = len(spot_returns)
    if est_window < 2 or est_window >= n_days:
        raise ValueError(f"need 2 <= est_window < n_days, got {est_window} vs {n_days}")
    if rebalance_every < 1:
        raise ValueError(f"rebalance_every must be >= 1, got {rebalance_every}")
    cols = list(spot_returns.columns)
    cost_rate = (
        pd.Series(float(cost_bps), index=cols)
        if np.isscalar(cost_bps)
        else pd.Series(cost_bps).reindex(cols)
    ) * 1e-4
    if cost_rate.isna().any() or (cost_rate < 0).any():
        raise ValueError("cost_bps must be non-negative and cover all assets")

    total = spot_returns + carry_returns
    targets = np.full((n_days, len(cols)), np.nan)
    for t in range(est_window - 1, n_days):
        if (t - est_window + 1) % rebalance_every == 0:
            w = pd.Series(weight_func(total.iloc[: t + 1])).reindex(cols)
            if w.isna().any():
                raise ValueError("weight_func returned NaN or missing assets")
            targets[t] = w.to_numpy(dtype=float)
    run = run_fast(
        pd.DataFrame(np.expm1(spot_returns.to_numpy(dtype=float)), index=spot_returns.index,
                     columns=cols),
        targets,
        fill=NextBarFill.from_execution(execution, auction_booked="next"),
        cost=HalfSpreadCost(np.asarray(cost_rate, dtype=np.float64)),
        asset_carry=carry_returns.to_numpy(dtype=float),
    )
    led = run.ledger.iloc[est_window:]
    ledger = pd.DataFrame(
        {"spot_pnl": led["gross_ret"], "carry_pnl": led["carry"], "cost": led["cost"],
         "net": led["net_ret"], "turnover": led["turnover"]},
        index=pd.DatetimeIndex(led.index),
    )
    weights = run.weights.iloc[est_window:].set_axis(ledger.index)
    return BacktestResult(ledger=ledger, weights=weights)


# ---------------------------------------------------------------------------
# Base-currency conversion (GBP reporting for a London desk)
# ---------------------------------------------------------------------------


def base_conversion_returns(
    base_spot: pd.Series,
    rates: pd.DataFrame,
    base: str = "GBP",
    quote: str = "USD",
    dt: float = DT,
) -> pd.Series:
    """Total log return of the QUOTE currency measured in the new BASE currency.

    Given the BASEQUOTE spot (e.g. GBPUSD = USD per 1 GBP) and deposit rates,
    the daily total log return of holding QUOTE cash from the perspective of a
    BASE investor is

        c_t = -log(S_t / S_{t-1}) + (i_quote,{t-1} - i_base,{t-1}) * dt

    (spot leg: QUOTE appreciates when BASEQUOTE falls; carry leg uses
    previous-close rates, same convention as the return panel).

    Returns
    -------
    pd.Series
        Conversion increment ``c_t``; add it to every asset's QUOTE-base
        total log return to re-express it in the new base.
    """
    if (base_spot <= 0).any():
        raise ValueError("base_spot must be strictly positive")
    for col in (base, quote):
        if col not in rates.columns:
            raise ValueError(f"rates missing column {col!r}")
    spot_leg = -np.log(base_spot).diff()
    carry_leg = ((rates[quote] - rates[base]) * dt).shift(1)
    out = (spot_leg + carry_leg).iloc[1:]
    out.name = f"{quote}_in_{base}"
    return out


def convert_base(
    returns: pd.DataFrame | pd.Series,
    conversion: pd.Series,
) -> pd.DataFrame | pd.Series:
    """Re-express log total returns in a new base currency.

    Adds the conversion increment (from :func:`base_conversion_returns`) to
    every column.  Identity (tested): switching base changes each asset's log
    return by EXACTLY the old base currency's own total return in the new
    base; consequently the weighted log-return series of a dollar-neutral
    portfolio (sum of weights = 0) is invariant to the base choice.
    """
    conv = conversion.reindex(
        returns.index if isinstance(returns, (pd.Series, pd.DataFrame)) else None
    )
    if conv.isna().any():
        raise ValueError("conversion series must cover the returns index")
    if isinstance(returns, pd.DataFrame):
        return returns.add(conv, axis=0)
    return returns + conv
