"""Walk-forward backtest of the regime strategy, on the shared ``qp_bt`` core.

Execution convention (the same in every strategy project of this repo):

* Signals use data up to the close of day t. Orders fill at a modelled
  t+1 price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a
  close-auction variant fills at p_t and pays an auction cost.
* The regime label decided at the close of t sets the target book w_t. The
  period arithmetic is :func:`qp_bt.run_fast`: the book is traded to w_t as
  weights of the post-trade NAV during day t+1, costs come out of the trade,
  and on days with no rebalance the book drifts with prices (it is not
  reset to w for free). R is the simple return of the currency against USD.
* Carry is the deposit-rate differential over USD at close t, ACT/365 over
  the calendar gap to the next close, earned on the book held overnight
  (``asset_carry``; the FX roll is paid on what is held at the roll).
* Costs: half-spread per leg on traded notional (pips converted with each
  pair's pip size and quote level at the decision close,
  :class:`qp_bt.HalfSpreadCost`), plus ``auction_cost_bp`` per unit of traded
  notional in close-auction mode.

Ledger identity (tested exactly): net = spot + carry - cost, row by row.

Migration (Round 12, W24b). Before, the book was reset to its target every
day for free, the fill day was split additively between the old and new
books (missing the cross term), costs were charged on the pre-trade NAV,
and carry was time-weighted f : (1 - f) between the two books. Before/after
numbers are in the handback; hand fixtures are in ``tests/test_backtest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qp_bt import HalfSpreadCost, NextBarFill, run_fast
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
    weights : DataFrame (dates x currencies): the book held at each close
        (after the day's fill and drift), as weights of that close's NAV.
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

    reg_vals = regimes.reindex(idx).to_numpy(dtype=object)
    # half-spread of each leg, as a fraction of notional, at each close
    levels = quote_levels(returns).to_numpy()
    half_ref = half_spreads(currencies, cfg.spread_pips).to_numpy()
    ref_lvl = np.array([QUOTES[c][2] if c in QUOTES else 1.0 for c in currencies])
    half_all = np.asarray(half_ref[None, :] * ref_lvl[None, :] / levels, dtype=np.float64)
    rates_np = rates[currencies].to_numpy(dtype=float)
    usd_np = rates["USD"].to_numpy(dtype=float)
    ret_np = returns.to_numpy(dtype=float)
    if isinstance(idx, pd.DatetimeIndex):
        gaps = np.r_[1.0, (idx[1:] - idx[:-1]).days.to_numpy(dtype=float)]
    else:
        gaps = np.ones(len(idx))
    # carry of row t: the differential at close t-1 over the gap to close t
    carry = np.zeros_like(ret_np)
    carry[1:] = (rates_np[:-1] - usd_np[:-1, None]) * (gaps[1:, None] / 365.0)

    p = len(currencies)
    targets = np.full((len(idx), p), np.nan)
    labels: list[str] = []
    w_prev = np.zeros(p)
    last_regime: str | None = None
    days_since_rebal = 0
    for t in trade_days:
        regime = reg_vals[t - 1]
        if not isinstance(regime, str):
            regime = None  # no signal yet at the decision date -> stay flat
        if regime is None:
            w = np.zeros(p)
        elif (regime != last_regime or days_since_rebal >= cfg.rebalance_every
              or not np.any(w_prev)):
            cov = np.cov(ret_np[t - cfg.cov_window : t].T, ddof=1).reshape(p, p)
            w = regime_weights(regime, currencies, rates.iloc[t - 1], cov,
                               cfg).to_numpy(dtype=float)
            days_since_rebal = 0
        else:
            w = w_prev
        if w is not w_prev:
            targets[t - 1] = w
        labels.append(regime if regime is not None else "flat")
        w_prev = w
        last_regime = regime
        days_since_rebal += 1

    run = run_fast(
        pd.DataFrame(np.expm1(ret_np), index=idx, columns=currencies),
        targets,
        fill=NextBarFill.from_execution(ex, auction_booked="next"),
        cost=HalfSpreadCost(half_all),
        asset_carry=carry,
    )
    led = run.ledger.iloc[trade_days[0]:]
    ledger = pd.DataFrame(
        {"regime": labels, "spot": led["gross_ret"].to_numpy(), "carry": led["carry"].to_numpy(),
         "cost": led["cost"].to_numpy(), "net": led["net_ret"].to_numpy(),
         "turnover": led["turnover"].to_numpy()},
        index=pd.Index(idx[trade_days[0]:], name="date"),
    )
    weights = run.weights.iloc[trade_days[0]:].set_axis(ledger.index)
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
