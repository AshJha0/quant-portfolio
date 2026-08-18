"""Regime-conditional FX baskets with carry accrual, vol targeting, pip costs.

Playbook (weights are on currency-vs-USD legs; +w = long the currency
against USD):

* ``risk_on``     : rank-carry basket — long the top-n yielders, short
                    the bottom-n, dollar-neutral, vol-targeted.
* ``risk_off``    : carry is CUT; long safe havens (JPY, CHF) against
                    the risk block (antipodeans + EM — the legs that
                    unwind hardest), vol-targeted.
* ``usd_squeeze`` : long USD against everything (short every currency
                    leg equally), vol-targeted.

P&L conventions
---------------
Daily gross return = w . (spot log return) + w . (r_ccy - r_USD)/252
(carry accrual, ACT/252 simplification).  Transaction costs are charged
in PIPS on turnover: cost_t = sum_i |w_t,i - w_{t-1},i| * half_spread_i,
with half_spread_i = SPREAD_PIPS[i] * 1e-4 (1 pip ~ 1 bp of notional on
a quote near 1.0; EM legs wider).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.synthetic import PIP_FRACTION, SPREAD_PIPS

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
    weights: pd.Series, rates_row: pd.Series, usd_rate: float
) -> float:
    """One day of carry: sum_i w_i * (r_i - r_USD) / 252.

    Parameters
    ----------
    weights : currency-vs-USD weights.
    rates_row : annualised deposit rates (indexed by currency).
    usd_rate : annualised USD deposit rate.
    """
    diffs = rates_row.reindex(weights.index) - usd_rate
    return float((weights * diffs).sum() / TRADING_DAYS)


def transaction_cost(
    new_weights: pd.Series,
    old_weights: pd.Series,
    spread_pips: dict[str, float] | None = None,
) -> float:
    """Pip cost of moving the book: sum_i |dw_i| * half_spread_i.

    half_spread_i = spread_pips[i] * 1e-4 (fraction of notional).
    Currencies missing from the table default to 2 pips.
    """
    table = spread_pips if spread_pips is not None else SPREAD_PIPS
    dw = (new_weights - old_weights.reindex(new_weights.index).fillna(0.0)).abs()
    half = pd.Series(
        {c: table.get(c, 2.0) * PIP_FRACTION for c in new_weights.index}
    )
    return float((dw * half).sum())
