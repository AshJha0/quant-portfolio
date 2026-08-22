"""Walk-forward backtest of the regime strategy, with exact ledger.

Timing convention (no lookahead, test-enforced end to end)
----------------------------------------------------------
* Features at ``t`` use prices up to and including ``t`` (expanding
  z-scores).
* The HMM used at ``t`` was last refitted on features strictly BEFORE its
  refit date and never sees data after ``t``.
* The filtered probability at ``t`` uses features up to and including ``t``.
* The weight ``w_t`` decided at the close of ``t`` earns the return from
  ``t`` to ``t+1`` (``r_{t+1}``).
* Transaction costs: ``cost_t = cost_bps/1e4 * |w_t - w_{t-1}|``, charged
  on the day the trade is made.

Ledger columns (one row per traded day): ``weight``, ``gross_ret``,
``cost``, ``net_ret``, ``equity`` with
``net_ret_t = weight_{t-1} * r_t - cost_t`` and
``equity_t = equity_{t-1} * (1 + net_ret_t)``.  Simple returns are used for
compounding; the input panel's log-returns are converted internally.

Benchmarks: buy-and-hold of the equal-weight index, and a classic 200-day
moving-average timing rule (long above the MA, cash below — decided at
``t``, applied to ``t+1``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .detection import expanding_fit_detect
from .features import build_features
from .strategy import build_weights

TRADING_DAYS: int = 252

__all__ = ["BacktestResult", "run_ledger", "ma_timing_weights", "walk_forward_backtest", "summary_stats"]


@dataclass(frozen=True)
class BacktestResult:
    """Everything a risk report needs.

    Attributes
    ----------
    ledger : pd.DataFrame
        Strategy ledger (weight, gross_ret, cost, net_ret, equity).
    benchmark : pd.DataFrame
        Buy-and-hold ledger over the same dates.
    ma_rule : pd.DataFrame
        200d-MA timing ledger over the same dates.
    detection : pd.DataFrame
        Filtered probabilities and regime labels used for trading.
    features : pd.DataFrame
        Point-in-time feature table.
    """

    ledger: pd.DataFrame
    benchmark: pd.DataFrame
    ma_rule: pd.DataFrame
    detection: pd.DataFrame
    features: pd.DataFrame


def run_ledger(
    weights: pd.Series,
    simple_returns: pd.Series,
    cost_bps: float = 5.0,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    """Exact ledger: ``w_t`` earns ``r_{t+1}``; costs on trade days.

    Parameters
    ----------
    weights : pd.Series
        Weight decided at the close of each date (aligned to dates ``t``).
    simple_returns : pd.Series
        SIMPLE daily returns of the traded factor; must cover
        ``weights.index`` shifted one day forward.
    cost_bps : float
        One-way transaction cost in basis points of traded notional.
    initial_equity : float
        Starting equity.

    Returns
    -------
    pd.DataFrame
        Indexed by P&L dates (each weight date's NEXT trading day), columns
        ``weight`` (the weight earning that day's return), ``gross_ret``,
        ``cost``, ``net_ret``, ``equity``.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if len(weights) < 2:
        raise ValueError("need at least 2 weight observations")
    w = weights.astype(float)
    dw = w.diff()
    dw.iloc[0] = w.iloc[0]  # entering from flat
    cost_rate = cost_bps / 1e4

    ret_pos = simple_returns.index.get_indexer(w.index)
    if (ret_pos < 0).any():
        raise ValueError("weights index not contained in returns index")
    if ret_pos[-1] + 1 >= len(simple_returns):
        # last weight has no next-day return: drop it
        w = w.iloc[:-1]
        dw = dw.iloc[:-1]
        ret_pos = ret_pos[:-1]
    next_ret = simple_returns.to_numpy()[ret_pos + 1]
    pnl_dates = simple_returns.index[ret_pos + 1]

    gross = w.to_numpy() * next_ret
    cost = cost_rate * np.abs(dw.to_numpy())
    net = gross - cost
    equity = initial_equity * np.cumprod(1.0 + net)
    return pd.DataFrame(
        {
            "weight": w.to_numpy(),
            "gross_ret": gross,
            "cost": cost,
            "net_ret": net,
            "equity": equity,
        },
        index=pnl_dates,
    )


def ma_timing_weights(prices: pd.DataFrame, ma_window: int = 200) -> pd.Series:
    """200d-MA timing rule: weight 1 when the index closes above its MA, else 0.

    Decided at the close of ``t`` from prices up to ``t`` (causal).
    """
    if ma_window < 2:
        raise ValueError(f"ma_window must be >= 2, got {ma_window}")
    norm = prices / prices.iloc[0]
    idx = norm.mean(axis=1)
    ma = idx.rolling(ma_window).mean()
    return (idx > ma).astype(float).rename("weight")


def walk_forward_backtest(
    prices: pd.DataFrame,
    n_states: int = 3,
    min_train: int = 252,
    refit_every: int = 63,
    cost_bps: float = 5.0,
    seed: int = 0,
    enter: float = 0.70,
    exit_: float = 0.30,
    target_vol: float | None = 0.10,
    n_pca: int | None = 3,
    feature_kwargs: dict | None = None,
    detect_kwargs: dict | None = None,
) -> BacktestResult:
    """End-to-end walk-forward backtest on a price panel.

    Pipeline: point-in-time features -> expanding-window HMM refits every
    ``refit_every`` days -> ONLINE filtered probabilities -> hysteresis +
    vol-targeted weights -> next-day-return ledger net of costs.
    Benchmarks (buy-and-hold, 200d-MA rule) are run over the SAME dates with
    the SAME cost model.

    Parameters
    ----------
    prices : pd.DataFrame
        (T x N) daily price panel.
    n_states, min_train, refit_every, seed : detection settings.
    n_pca : int or None
        Number of principal components the HMM observes (None = raw
        features).  PCA is refitted on each expanding training window.
    cost_bps : float
        One-way cost in bps for strategy AND benchmarks.
    enter, exit_ : hysteresis band on ``p_bear``.
    target_vol : float or None
        Annualised vol target (None disables vol targeting).
    feature_kwargs, detect_kwargs : dict
        Overrides forwarded to :func:`build_features` /
        :func:`expanding_fit_detect`.

    Returns
    -------
    BacktestResult
    """
    features = build_features(prices, **(feature_kwargs or {}))
    detection = expanding_fit_detect(
        features,
        n_states=n_states,
        min_train=min_train,
        refit_every=refit_every,
        seed=seed,
        n_pca=n_pca,
        **(detect_kwargs or {}),
    )
    asset_log_ret = np.log(prices / prices.shift(1)).iloc[1:]
    log_ret = asset_log_ret.mean(axis=1)
    simple_ret = np.expm1(asset_log_ret).mean(axis=1)

    weights = build_weights(
        detection,
        log_ret,
        enter=enter,
        exit_=exit_,
        target_vol=target_vol,
    )
    ledger = run_ledger(weights, simple_ret, cost_bps=cost_bps)

    bh_weights = pd.Series(1.0, index=weights.index, name="weight")
    benchmark = run_ledger(bh_weights, simple_ret, cost_bps=cost_bps)

    ma_w = ma_timing_weights(prices).reindex(weights.index).fillna(0.0)
    ma_rule = run_ledger(ma_w, simple_ret, cost_bps=cost_bps)

    return BacktestResult(
        ledger=ledger,
        benchmark=benchmark,
        ma_rule=ma_rule,
        detection=detection,
        features=features,
    )


def summary_stats(ledger: pd.DataFrame) -> dict[str, float]:
    """Headline stats of a ledger: CAGR, ann. vol, Sharpe, max drawdown, turnover cost.

    Sharpe uses net daily returns, zero risk-free, annualised with sqrt(252).
    Max drawdown is on the compounded equity curve (positive number).
    """
    net = ledger["net_ret"].to_numpy()
    equity = ledger["equity"].to_numpy()
    n_yrs = len(net) / TRADING_DAYS
    cagr = float(equity[-1] ** (1.0 / n_yrs) - 1.0) if n_yrs > 0 else np.nan
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS)) if net.std(ddof=1) > 0 else np.nan
    peak = np.maximum.accumulate(equity)
    mdd = float((1.0 - equity / peak).max())
    return {
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "total_cost": float(ledger["cost"].sum()),
        "final_equity": float(equity[-1]),
    }
