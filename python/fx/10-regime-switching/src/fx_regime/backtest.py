"""Walk-forward backtest engine with an exact ledger and no lookahead.

Execution convention (the same in every strategy project of this repo):

* Signals use data up to the close of day t. Orders fill at a modelled
  t+1 price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a
  close-auction variant fills at p_t and pays an auction cost.
* Here the regime label decided at the close of t sets the book w_t.
  Over day t+1 the old book w_{t-1} earns the move from p_t to the fill
  price, ``p_fill / p_t - 1 = f R``, and the new book earns the rest,
  ``p_{t+1} / p_fill - 1 = (1 + R) / (1 + f R) - 1``, where R is the
  simple return of the currency against USD over day t+1. f = 1 is a fill
  at the next close.
* Carry accrues ACT/365 over the calendar gap between the two closes,
  split between the two books in the same proportion f : (1 - f).
* Costs: half-spread per leg on turnover (pips converted with each pair's
  pip size and quote level), plus ``auction_cost_bp`` per unit of traded
  notional in close-auction mode.

Ledger identity (tested exactly): net = spot + carry - cost, row by row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qp_core.execution import ExecutionConfig

from .data.synthetic import QUOTES, quote_levels
from .strategy import (
    StrategyConfig,
    half_spreads,
    regime_weights,
)

TRADING_DAYS = 252



@dataclass
class BacktestResult:
    """Backtest output.

    Attributes
    ----------
    ledger : DataFrame indexed by date with columns
        ``regime`` (label decided at the previous close, whose book is
        filled during this day), ``spot``, ``carry``, ``cost``,
        ``net`` (= spot + carry - cost), ``turnover``.
    weights : DataFrame (dates x currencies): the book decided at the
        previous close and filled during each day.
    """

    ledger: pd.DataFrame
    weights: pd.DataFrame

    @property
    def net(self) -> pd.Series:
        return self.ledger["net"]


def run_backtest(
    returns: pd.DataFrame,
    deposit_rates: pd.DataFrame,
    regimes: pd.Series,
    config: StrategyConfig | None = None,
    execution: ExecutionConfig | None = None,
) -> BacktestResult:
    """Run the regime-conditional strategy walk-forward.

    Parameters
    ----------
    returns : DataFrame (T x p)
        Daily log returns of currency vs USD.
    deposit_rates : DataFrame
        Annualised deposit rates with a ``USD`` column, same index.
    regimes : Series of labels indexed by a subset of ``returns.index``.
        The label at date t is decided at the close of t; its book is
        filled during day t+1 under ``execution``.
    config : StrategyConfig
    execution : ExecutionConfig (default: next-period fill, f = 0.5)

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError
        If indices are incompatible, inputs are not finite, or the sample
        is too short.
    """
    cfg = config or StrategyConfig()
    ex = execution or ExecutionConfig()
    if "USD" not in deposit_rates.columns:
        raise ValueError("deposit_rates must include a USD column")
    if not regimes.index.isin(returns.index).all():
        raise ValueError("regimes index must be a subset of returns index")
    if not np.all(np.isfinite(returns.to_numpy(dtype=float))):
        raise ValueError("returns contain NaN or infinite values")
    currencies = list(returns.columns)
    rates = deposit_rates.reindex(returns.index).ffill()
    missing = [c for c in currencies if c not in rates.columns]
    if missing:
        raise ValueError(f"deposit_rates missing currencies {missing}")

    idx = returns.index
    pos = idx.get_indexer(regimes.index)
    first_signal = int(pos[0])
    # need cov_window history before the first tradable day
    start = max(first_signal, cfg.cov_window)
    trade_days = [t for t in range(start + 1, len(idx))]
    if len(trade_days) < 2:
        raise ValueError("sample too short to trade after warm-up")

    reg_full = regimes.reindex(idx)
    reg_vals = reg_full.to_numpy(dtype=object)
    # half-spread of each leg, as a fraction of notional, at each close
    levels = quote_levels(returns).to_numpy()
    half_ref = half_spreads(currencies, cfg.spread_pips).to_numpy()
    ref_lvl = np.array([QUOTES[c][2] if c in QUOTES else 1.0 for c in currencies])
    half_all = half_ref[None, :] * ref_lvl[None, :] / levels  # pips*pip/level
    rates_np = rates[currencies].to_numpy(dtype=float)
    usd_np = rates["USD"].to_numpy(dtype=float)

    dates = []
    rows = []
    w_rows = []
    p = len(currencies)
    w_prev = np.zeros(p)
    last_regime: str | None = None
    days_since_rebal = 0

    ret_np = returns.to_numpy(dtype=float)
    simple = np.expm1(ret_np)
    f = 0.0 if ex.mode == "close_auction" else ex.fill_fraction
    if isinstance(idx, pd.DatetimeIndex):
        gaps = np.r_[1.0, (idx[1:] - idx[:-1]).days.to_numpy(dtype=float)]
    else:
        gaps = np.ones(len(idx))

    for t in trade_days:
        regime = reg_vals[t - 1]
        if not isinstance(regime, str):
            # no signal yet at the decision date -> stay flat
            regime = None
        if regime is None:
            w = np.zeros(p)
        elif (
            regime != last_regime
            or days_since_rebal >= cfg.rebalance_every
            or not np.any(w_prev)
        ):
            cov = np.cov(
                ret_np[t - cfg.cov_window : t].T, ddof=1
            ).reshape(p, p)
            w = regime_weights(
                regime, currencies, rates.iloc[t - 1], cov, cfg
            ).to_numpy(dtype=float)
            days_since_rebal = 0
        else:
            w = w_prev
        dw = np.abs(w - w_prev)
        turnover = float(dw.sum())
        cost = float(dw @ half_all[t - 1])
        if ex.mode == "close_auction":
            cost += ex.auction_cost_bp * 1e-4 * turnover
        R = simple[t]
        old_leg = f * R
        new_leg = (1.0 + R) / (1.0 + f * R) - 1.0
        spot = float(w_prev @ old_leg + w @ new_leg)
        diffs = (rates_np[t - 1] - usd_np[t - 1]) * (gaps[t] / 365.0)
        carry = float(f * (w_prev @ diffs) + (1.0 - f) * (w @ diffs))
        rows.append(
            {
                "regime": regime if regime is not None else "flat",
                "spot": spot,
                "carry": carry,
                "cost": cost,
                "net": spot + carry - cost,
                "turnover": turnover,
            }
        )
        dates.append(idx[t])
        w_rows.append(w)
        w_prev = w
        last_regime = regime
        days_since_rebal += 1

    ledger = pd.DataFrame(rows, index=pd.Index(dates, name="date"))
    weights = pd.DataFrame(np.array(w_rows), index=ledger.index, columns=currencies)
    return BacktestResult(ledger=ledger, weights=weights)


def static_carry_regimes(index: pd.Index) -> pd.Series:
    """Always-risk-on regime path: the static carry benchmark."""
    return pd.Series("risk_on", index=index, name="regime")


def oracle_regimes(
    index: pd.Index, true_states: np.ndarray, state_names: tuple[str, ...]
) -> pd.Series:
    """Regime path of an oracle that KNOWS the true hidden state.

    The oracle observes the true state at the close of each day and
    positions for the next day, same execution convention as the
    filtered strategy, but zero detection lag.  Its edge over the
    filtered strategy is therefore exactly the price of detection lag
    plus classification error.

    Parameters
    ----------
    index : dates of the sample.
    true_states : (T,) int states aligned to ``index``.
    state_names : label per state index.
    """
    if len(index) != len(true_states):
        raise ValueError("index and true_states must have equal length")
    return pd.Series(
        [state_names[s] for s in true_states], index=index, name="regime"
    )
