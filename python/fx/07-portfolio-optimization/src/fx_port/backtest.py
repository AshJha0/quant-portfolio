"""Walk-forward backtesting for currency portfolios.

Design rules (all tested):

* **No lookahead** — the weight function receives history strictly up to and
  including the rebalance date *t*; new weights take effect from *t+1*.
  Carry accrual inside the returns already uses previous-close rates (see
  :mod:`fx_port.returns_est`).
* **Carry in the P&L** — the ledger splits every day's P&L into a spot leg
  and a carry-accrual leg; their sum plus costs is the net return.
* **Transaction costs in pips** — FX costs quote as a spread in pips;
  :func:`pips_to_bps` converts ``pips × pip_size / spot`` into basis points
  of notional.  The engine charges ``cost_rate × turnover`` on the day new
  weights take effect, with turnover = sum |w_new - w_old| (weights are
  treated as reset to target at each rebalance; intra-period drift of
  long-short FX weights is second-order and documented as a simplification).
* **Base-currency reporting** — helpers convert USD-base log total returns to
  any base (the GBP option for a London desk).  For LOG returns the change of
  base adds the SAME conversion term to every asset, so a dollar-neutral
  portfolio's log-return series is EXACTLY invariant to the base currency
  (the common term multiplies sum(w) = 0); a net-long book shifts by the base
  currency's own total return.  Both identities are unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

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
        Daily rows: ``spot_pnl``, ``carry_pnl``, ``cost``, ``net``
        (= spot + carry - cost) and ``turnover``.
    weights : pd.DataFrame
        Target weights as applied on each day (post-lag).
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
) -> BacktestResult:
    """Walk-forward backtest with rebalancing, pip costs and carry accrual.

    Parameters
    ----------
    spot_returns, carry_returns : pd.DataFrame
        Daily spot log returns and carry accrual per asset (same shape;
        their sum is the total return).  Obtain them from
        :func:`fx_port.returns_est.total_log_returns`.
    weight_func : callable
        ``weight_func(history)`` -> weights Series, where ``history`` is the
        TOTAL return panel up to and including the rebalance date.  Called
        first on row ``est_window - 1`` (i.e. after ``est_window`` days of
        history) and then every ``rebalance_every`` days.
    est_window : int
        Minimum history before the first allocation (>= 2).
    rebalance_every : int
        Days between rebalances (>= 1).
    cost_bps : float or pd.Series
        One-way transaction cost per unit of turnover, in bps of notional,
        scalar or per-asset.

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
        raise ValueError(
            f"need 2 <= est_window < n_days, got {est_window} vs {n_days}"
        )
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
    spot_np, carry_np = spot_returns.to_numpy(), carry_returns.to_numpy()
    cost_np = cost_rate.to_numpy()

    w_current = np.zeros(len(cols))
    pending: tuple[int, np.ndarray] | None = None
    rows, w_rows, dates = [], [], []
    rebalance_days = set(range(est_window - 1, n_days, rebalance_every))
    for t in range(est_window - 1, n_days):
        if pending is not None and pending[0] == t:
            w_new = pending[1]
            turnover = float(np.abs(w_new - w_current).sum())
            cost = float(np.abs(w_new - w_current) @ cost_np)
            w_current = w_new
            pending = None
        else:
            turnover, cost = 0.0, 0.0
        if t > est_window - 1:  # P&L accrues only after the first allocation
            spot_pnl = float(w_current @ spot_np[t])
            carry_pnl = float(w_current @ carry_np[t])
            rows.append(
                [spot_pnl, carry_pnl, cost, spot_pnl + carry_pnl - cost, turnover]
            )
            w_rows.append(w_current.copy())
            dates.append(spot_returns.index[t])
        if t in rebalance_days:
            hist = total.iloc[: t + 1]
            w = pd.Series(weight_func(hist)).reindex(cols)
            if w.isna().any():
                raise ValueError("weight_func returned NaN or missing assets")
            pending = (t + 1, w.to_numpy(dtype=float))
    ledger = pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(dates),
        columns=["spot_pnl", "carry_pnl", "cost", "net", "turnover"],
    )
    weights = pd.DataFrame(w_rows, index=ledger.index, columns=cols)
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
