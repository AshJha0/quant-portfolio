"""Regime-conditional FX baskets with carry accrual, vol targeting, pip costs.

Playbook (weights are on currency-vs-USD legs; +w = long the currency
against USD):

* ``risk_on``     : rank-carry basket, long the top-n yielders, short
                    the bottom-n, dollar-neutral, vol-targeted.
* ``risk_off``    : carry is CUT; long safe havens (JPY, CHF) against
                    the risk block (antipodeans + EM, the legs that
                    unwind hardest), vol-targeted.
* ``usd_squeeze`` : long USD against everything (short every currency
                    leg equally), vol-targeted.

P&L conventions
---------------
Spot P&L is w . (simple return of the currency vs USD) under the fill
convention of :mod:`fx_regime.backtest`. Carry accrues ACT/365 on the
calendar gap between closes: w . (r_ccy - r_USD) * days / 365. Transaction
costs are charged on turnover: cost_t = sum_i |w_t,i - w_{t-1},i| *
half_spread_i, where half_spread_i = pips_i * pip_size_i / quote_level_i
(pip size 0.01 for USDJPY, 0.0001 otherwise; quote level is the pair's
price, so a pip on USDMXN at 17.5 is worth 1/17.5 of a pip on EURUSD).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import warnings

import numpy as np
import pandas as pd

from .data.synthetic import SPREAD_PIPS, half_spread_fraction

TRADING_DAYS = 252


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy controls.

    Attributes
    ----------
    target_vol : float
        Annualised ex-ante vol target of the basket (e.g. 0.10 = 10%).
    max_leverage : float
        Cap on the gross scaling multiplier from vol targeting.
    cov_window : int
        Rolling window (days) for the ex-ante covariance estimate.
    n_carry_long, n_carry_short : int
        Rank-carry basket sizes for the risk_on book.
    haven_currencies, risk_proxy_currencies : tuple of str
        Legs of the risk_off book.
    rebalance_every : int
        Days between scheduled weight recomputes (a regime change always
        forces a rebalance).
    vol_floor : float
        Annualised floor on ex-ante basket vol before scaling (guards
        against pegged / degenerate covariance).
    spread_pips : mapping currency -> half-spread in pips.
    """

    target_vol: float = 0.10
    max_leverage: float = 4.0
    cov_window: int = 63
    n_carry_long: int = 3
    n_carry_short: int = 3
    haven_currencies: tuple[str, ...] = ("JPY", "CHF")
    risk_proxy_currencies: tuple[str, ...] = ("AUD", "NZD", "MXN", "ZAR", "BRL")
    rebalance_every: int = 5
    vol_floor: float = 0.01
    spread_pips: dict = field(default_factory=lambda: dict(SPREAD_PIPS))

    def __post_init__(self) -> None:
        if self.target_vol <= 0:
            raise ValueError("target_vol must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.cov_window < 10:
            raise ValueError("cov_window must be >= 10")


def base_weights(
    regime: str,
    currencies: list[str],
    rates_row: pd.Series,
    config: StrategyConfig,
) -> pd.Series:
    """Unscaled regime book (before vol targeting).

    Parameters
    ----------
    regime : {"risk_on", "risk_off", "usd_squeeze"}
    currencies : tradeable currency-vs-USD legs.
    rates_row : annualised deposit rates as of the decision date (used
        for carry ranking in risk_on).
    config : StrategyConfig

    Returns
    -------
    Series of weights indexed by ``currencies``.

    Raises
    ------
    ValueError
        On an unknown regime label.
    """
    w = pd.Series(0.0, index=currencies)
    if regime == "risk_on":
        n_l, n_s = config.n_carry_long, config.n_carry_short
        if n_l + n_s > len(currencies):
            raise ValueError("carry basket sizes exceed universe")
        r = rates_row.reindex(currencies)
        order = r.sort_values(kind="stable").index
        w[order[-n_l:]] = 1.0 / n_l
        w[order[:n_s]] = -1.0 / n_s
    elif regime == "risk_off":
        havens = [c for c in config.haven_currencies if c in currencies]
        risks = [c for c in config.risk_proxy_currencies if c in currencies]
        if not havens or not risks:
            raise ValueError("risk_off book needs haven and risk legs")
        w[havens] = 1.0 / len(havens)
        w[risks] = -1.0 / len(risks)
    elif regime == "usd_squeeze":
        w[:] = -1.0 / len(currencies)
    else:
        raise ValueError(f"unknown regime {regime!r}")
    return w


def vol_target_scale(
    weights: pd.Series,
    cov_daily: np.ndarray,
    target_vol: float,
    max_leverage: float = 4.0,
    vol_floor: float = 0.01,
) -> float:
    """Scaling multiplier so the basket's ex-ante annualised vol hits target.

    scale = target_vol / max(sqrt(252 * w' C w), vol_floor), capped at
    ``max_leverage``.

    Parameters
    ----------
    weights : unscaled weights.
    cov_daily : (p, p) DAILY covariance estimate aligned to weights.
    target_vol : annualised target.
    max_leverage : cap on the multiplier.
    vol_floor : annualised floor on the raw basket vol.
    """
    w = weights.to_numpy(dtype=float)
    var_daily = float(w @ cov_daily @ w)
    ann_vol = np.sqrt(max(var_daily, 0.0) * TRADING_DAYS)
    ann_vol = max(ann_vol, vol_floor)
    return min(target_vol / ann_vol, max_leverage)


def regime_weights(
    regime: str,
    currencies: list[str],
    rates_row: pd.Series,
    cov_daily: np.ndarray,
    config: StrategyConfig,
) -> pd.Series:
    """Vol-targeted regime book: :func:`base_weights` x :func:`vol_target_scale`."""
    w = base_weights(regime, currencies, rates_row, config)
    scale = vol_target_scale(
        w, cov_daily, config.target_vol, config.max_leverage, config.vol_floor
    )
    return w * scale


def carry_accrual(
    weights: pd.Series,
    rates_row: pd.Series,
    usd_rate: float,
    year_fraction: float | None = None,
) -> float:
    """Carry over one accrual period: sum_i w_i * (r_i - r_USD) * year_fraction.

    Parameters
    ----------
    weights : currency-vs-USD weights.
    rates_row : annualised deposit rates (indexed by currency).
    usd_rate : annualised USD deposit rate.
    year_fraction : accrual period in years. The backtest passes ACT/365,
        ``calendar_days / 365``, so a Friday-to-Monday period accrues three
        days. Omitting it falls back to the old ACT/252 ``1/252`` with a
        ``DeprecationWarning`` (removed next release).
    """
    if year_fraction is None:
        warnings.warn(
            "carry_accrual without year_fraction uses 1/252 (ACT/252); pass "
            "year_fraction=calendar_days/365 (ACT/365, the backtest convention)",
            DeprecationWarning,
            stacklevel=2,
        )
        year_fraction = 1.0 / TRADING_DAYS
    if not np.isfinite(year_fraction) or year_fraction < 0:
        raise ValueError(f"year_fraction must be finite and >= 0, got {year_fraction}")
    if not np.isfinite(usd_rate):
        raise ValueError("usd_rate must be finite")
    diffs = rates_row.reindex(weights.index) - usd_rate
    if not np.all(np.isfinite(diffs.to_numpy(dtype=float))):
        raise ValueError("rates_row must have a finite rate for every weighted currency")
    return float((weights * diffs).sum() * year_fraction)


def half_spreads(
    currencies,
    spread_pips: dict[str, float] | None = None,
    quote_levels: pd.Series | None = None,
) -> pd.Series:
    """Half-spread of each leg as a fraction of notional.

    ``spread_pips[c] * pip_size_c / quote_level_c``; currencies missing from
    the table default to 2 pips. ``quote_levels`` (current price of each
    leg's quoted pair) defaults to the reference levels.
    """
    table = spread_pips if spread_pips is not None else SPREAD_PIPS
    out = {}
    for c in currencies:
        lvl = None if quote_levels is None else float(quote_levels[c])
        out[c] = half_spread_fraction(c, table.get(c, 2.0), lvl)
    return pd.Series(out)


def transaction_cost(
    new_weights: pd.Series,
    old_weights: pd.Series,
    spread_pips: dict[str, float] | None = None,
    quote_levels: pd.Series | None = None,
) -> float:
    """Cost of moving the book: sum_i |dw_i| * half_spread_i (fraction of notional).

    half_spread_i = spread_pips[i] * pip_size_i / quote_level_i, see
    :func:`half_spreads`.
    """
    dw = (new_weights - old_weights.reindex(new_weights.index).fillna(0.0)).abs()
    half = half_spreads(new_weights.index, spread_pips, quote_levels)
    return float((dw * half).sum())
