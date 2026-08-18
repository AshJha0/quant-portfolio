"""Walk-forward backtest engine with an exact ledger and no lookahead.

Timing convention (strict, mutation-tested):

* The regime label for date t must be DECIDED at the close of t (it is
  produced by the filtered detector from data through t).
* The position held OVER day t+1 is computed from the regime and data
  known at the close of t (rates row t, covariance from returns through
  t).  Hence ``position[t+1] = f(information up to t)``.
* Day t+1 P&L = position . (spot return t+1) + position . carry(t) -
  cost of the trade done at the close of t.

Ledger identity (tested exactly): net = spot + carry - cost, row by row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import (
    StrategyConfig,
    carry_accrual,
    regime_weights,
    transaction_cost,
)

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Backtest output.

    Attributes
    ----------
    ledger : DataFrame indexed by date with columns
        ``regime`` (label driving the day's position), ``spot``,
        ``carry``, ``cost``, ``net`` (= spot + carry - cost), ``turnover``.
    weights : DataFrame (dates x currencies) — position held over each day.
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
) -> BacktestResult:
    """Run the regime-conditional strategy walk-forward.

    Parameters
    ----------
    returns : DataFrame (T x p)
        Daily log returns of currency vs USD.
    deposit_rates : DataFrame
        Annualised deposit rates with a ``USD`` column, same index.
    regimes : Series of labels indexed by a subset of ``returns.index``.
        The label at date t is treated as decided at the close of t and
        drives the position over the NEXT day.
    config : StrategyConfig

    Returns
    -------
    BacktestResult

    Raises
    ------
    ValueError
        If indices are incompatible or the sample is too short.
    """
    cfg = config or StrategyConfig()
    if "USD" not in deposit_rates.columns:
        raise ValueError("deposit_rates must include a USD column")
    if not regimes.index.isin(returns.index).all():
        raise ValueError("regimes index must be a subset of returns index")
    currencies = list(returns.columns)
    rates = deposit_rates.reindex(returns.index).ffill()

    idx = returns.index
    pos = idx.get_indexer(regimes.index)
    first_signal = int(pos[0])
    # need cov_window history before the first tradable day
    start = max(first_signal, cfg.cov_window)
    trade_days = [t for t in range(start + 1, len(idx))]
    if len(trade_days) < 2:
        raise ValueError("sample too short to trade after warm-up")

    reg_full = regimes.reindex(idx)

    dates = []
    rows = []
    w_rows = []
    w_prev = pd.Series(0.0, index=currencies)
    last_regime: str | None = None
    days_since_rebal = 0

    ret_np = returns.to_numpy(dtype=float)

    for t in trade_days:
        decision_date = idx[t - 1]
        regime = reg_full.iloc[t - 1]
        if not isinstance(regime, str):
            # no signal yet at the decision date -> stay flat
            regime = None
        if regime is None:
            w = pd.Series(0.0, index=currencies)
        elif (
            regime != last_regime
            or days_since_rebal >= cfg.rebalance_every
            or w_prev.abs().sum() == 0.0
        ):
            cov = np.cov(
                ret_np[t - cfg.cov_window : t].T, ddof=1
            ).reshape(len(currencies), len(currencies))
            w = regime_weights(
                regime, currencies, rates.loc[decision_date], cov, cfg
            )
            days_since_rebal = 0
        else:
            w = w_prev
        cost = transaction_cost(w, w_prev, cfg.spread_pips)
        turnover = float((w - w_prev).abs().sum())
        spot = float(w.to_numpy() @ ret_np[t])
        carry = carry_accrual(
            w, rates.loc[decision_date], float(rates.loc[decision_date, "USD"])
        )
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
    weights = pd.DataFrame(w_rows, index=ledger.index)
    return BacktestResult(ledger=ledger, weights=weights)


def static_carry_regimes(index: pd.Index) -> pd.Series:
    """Always-risk-on regime path: the static carry benchmark."""
    return pd.Series("risk_on", index=index, name="regime")


def oracle_regimes(
    index: pd.Index, true_states: np.ndarray, state_names: tuple[str, ...]
) -> pd.Series:
    """Regime path of an oracle that KNOWS the true hidden state.

    The oracle observes the true state at the close of each day and
    positions for the next day — same one-day execution delay as the
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
