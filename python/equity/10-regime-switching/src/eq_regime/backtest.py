"""Walk-forward backtest of the regime strategy, with exact ledger.

Timing convention (no lookahead, test-enforced end to end)
----------------------------------------------------------
* Features at ``t`` use prices up to and including ``t`` (expanding
  z-scores). The HMM used at ``t`` was last refitted on features strictly
  before its refit date. The filtered probability at ``t`` uses features up
  to and including ``t``.
* Signals use data up to the close of day t. Orders fill at a modelled t+1
  price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a close-auction
  variant fills at p_t and pays an auction cost. See :class:`ExecutionConfig`.
* With index return ``R`` over day ``t+1`` and fill fraction ``f``, the old
  weight ``w_{t-1}`` earns ``a = f R`` (close ``t`` to the fill) and the new
  weight ``w_t`` earns ``b = (1 + R) / (1 + f R) - 1`` (fill to close
  ``t+1``). Gross return of the day is ``(1 + w_{t-1} a)(1 + w_t b) - 1``.
* Costs: ``cost_bps/1e4 * |w_t - w_{t-1}|`` on the fill day, plus
  ``auction_cost_bp/1e4 * |w_t - w_{t-1}|`` in close-auction mode.
* Financing: the cash ``1 - h`` held overnight (``h`` the drifted weight at
  the previous close) earns ``cash_rate``; leverage pays
  ``cash_rate + borrow_spread_bps/1e4``, both ACT/252 per trading day.
* The arithmetic is the shared core ``qp_bt.run_fast``: the book drifts
  between closes and costs are paid out of the trade on the notional at the
  fill, so the weight after a fill is exactly the target.

Ledger columns (one row per P&L day): ``weight`` (``w_t``), ``prev_weight``,
``fill_ret`` (``p_fill / p_t - 1``), ``gross_ret``, ``cost``, ``financing``,
``net_ret``, ``equity`` with ``equity_t = equity_{t-1} (1 + net_ret_t)``.

Benchmarks: buy-and-hold of the equal-weight index, and a 200-day moving
average timing rule (long above the MA, cash below), both on the same
execution and cost model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from qp_bt import LinearCost, NextBarFill, run_fast
from qp_core.execution import ExecutionConfig
from qp_core.stats import Financing

from .detection import expanding_fit_detect
from .features import build_features
from .strategy import build_weights, vol_target_scale

TRADING_DAYS: int = 252

__all__ = [
    "BacktestResult",
    "ExecutionConfig",
    "backtest_from_detection",
    "ma_timing_weights",
    "random_timing_weights",
    "run_ledger",
    "summary_stats",
    "vol_target_weights",
    "walk_forward_backtest",
]



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
    random_timing : pd.DataFrame or None
        Ledger of the same regime weights circularly shifted by a random
        offset, times the same vol scale, rescaled to the strategy's mean
        weight (equal average exposure, random timing).
    vol_target : pd.DataFrame or None
        Ledger of plain vol targeting (weight 1 times the vol scale).
    """

    ledger: pd.DataFrame
    benchmark: pd.DataFrame
    ma_rule: pd.DataFrame
    detection: pd.DataFrame
    features: pd.DataFrame
    random_timing: pd.DataFrame | None = None
    vol_target: pd.DataFrame | None = None


def run_ledger(
    weights: pd.Series,
    simple_returns: pd.Series,
    cost_bps: float = 5.0,
    initial_equity: float = 1.0,
    execution: ExecutionConfig | None = None,
    cash_rate: float = 0.0,
    borrow_spread_bps: float = 50.0,
) -> pd.DataFrame:
    """Exact ledger of a weight path under :class:`ExecutionConfig`.

    Parameters
    ----------
    weights : pd.Series
        Weight decided at the close of each date ``t``.
    simple_returns : pd.Series
        Simple daily returns of the traded index; must contain every weight
        date and the following day.
    cost_bps : float
        One-way cost in bp of traded notional.
    initial_equity : float
        Starting equity.
    execution : ExecutionConfig, optional
        Fill timing; default ``ExecutionConfig()`` (next, f = 0.5).
    cash_rate : float
        Annual rate earned on ``1 - w`` when ``w < 1``.
    borrow_spread_bps : float
        Annual spread over ``cash_rate`` paid on ``w - 1`` when ``w > 1``.

    Returns
    -------
    pd.DataFrame indexed by P&L dates (each weight date's next trading day).
    """
    ex = execution if execution is not None else ExecutionConfig()
    for name, v in (("cost_bps", cost_bps), ("initial_equity", initial_equity),
                    ("cash_rate", cash_rate), ("borrow_spread_bps", borrow_spread_bps)):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v}")
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if borrow_spread_bps < 0:
        raise ValueError(f"borrow_spread_bps must be >= 0, got {borrow_spread_bps}")
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")
    if len(weights) < 2:
        raise ValueError("need at least 2 weight observations")
    w = weights.astype(float)
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("weights contain NaN or Inf")

    ret_pos = simple_returns.index.get_indexer(w.index)
    if (ret_pos < 0).any():
        raise ValueError("weights index not contained in returns index")
    if ret_pos[-1] + 1 >= len(simple_returns):
        w = w.iloc[:-1]
        ret_pos = ret_pos[:-1]
    r = simple_returns.to_numpy(dtype=float)[ret_pos + 1]
    if not np.isfinite(r).all():
        raise ValueError("returns contain NaN or Inf on traded dates")
    pnl_dates = simple_returns.index[ret_pos + 1]

    wn = w.to_numpy()
    wo = np.concatenate([[0.0], wn[:-1]])  # entering from flat
    f = ex.effective_fraction
    a = f * r
    # The P&L, fill, costs and financing come from the shared core (qp_bt):
    # period k earns the target decided at weight date k, with the book
    # drifting between closes and costs paid out of the trade. Financing is
    # on the cash held overnight, ACT/252 per trading day (RangeIndex).
    rr = np.zeros((len(wn) + 1, 1))
    rr[1:, 0] = r
    tgt = np.full((len(wn) + 1, 1), np.nan)
    tgt[:-1, 0] = wn
    fill = NextBarFill.from_execution(ex, auction_booked="next")
    fin = Financing(rate=cash_rate, borrow_spread_bp=borrow_spread_bps,
                    periods_per_year=float(TRADING_DAYS))
    run = run_fast(pd.DataFrame(rr), tgt, fill, LinearCost(cost_bps), financing=fin)
    led = run.ledger.iloc[1:]
    net = led["net_ret"].to_numpy()
    equity = initial_equity * np.cumprod(1.0 + net)
    return pd.DataFrame(
        {
            "weight": wn,
            "prev_weight": wo,
            "fill_ret": a,
            "gross_ret": led["gross_ret"].to_numpy(),
            "cost": led["cost"].to_numpy(),
            "financing": led["financing"].to_numpy(),
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


def random_timing_weights(
    weights: pd.Series, seed: int = 0, reference: pd.Series | None = None
) -> pd.Series:
    """Circularly shift a weight path by a random offset (random timing).

    The shift keeps the distribution and run lengths of the weights and
    destroys their alignment with the market. The offset is uniform on
    ``[n/10, 9n/10]``. When ``reference`` is given the shifted path is
    rescaled to the reference's mean weight (equal average exposure).
    """
    n = len(weights)
    if n < 10:
        raise ValueError("need at least 10 weights for random timing")
    rng = np.random.default_rng(seed)
    k = int(rng.integers(n // 10, 9 * n // 10 + 1))
    out = pd.Series(np.roll(weights.to_numpy(dtype=float), k), index=weights.index, name="weight")
    if reference is not None:
        m = float(out.mean())
        if m > 0:
            out = out * float(reference.mean()) / m
    return out


def vol_target_weights(
    index_log_returns: pd.Series,
    dates: pd.Index,
    target_vol: float = 0.10,
    window: int = 21,
    max_leverage: float = 1.5,
) -> pd.Series:
    """Plain vol targeting: weight ``min(target / trailing vol, max_leverage)``."""
    scale = vol_target_scale(
        index_log_returns, target_vol=target_vol, window=window, max_leverage=max_leverage
    )
    return scale.reindex(dates).clip(upper=max_leverage).rename("weight")


def backtest_from_detection(
    prices: pd.DataFrame,
    detection: pd.DataFrame,
    features: pd.DataFrame | None = None,
    cost_bps: float = 5.0,
    enter: float = 0.70,
    exit_: float = 0.30,
    target_vol: float | None = 0.10,
    min_duration: int = 1,
    execution: ExecutionConfig | None = None,
    cash_rate: float = 0.0,
    borrow_spread_bps: float = 50.0,
    random_seed: int = 0,
) -> BacktestResult:
    """Weights, ledgers and baselines from an existing detection table.

    Lets several weight variants share one (expensive) walk-forward
    detection. All ledgers use the same dates, execution and cost model.
    """
    asset_log_ret = cast(pd.DataFrame, np.log(prices / prices.shift(1))).iloc[1:]
    log_ret = asset_log_ret.mean(axis=1)
    simple_ret = cast(pd.DataFrame, np.expm1(asset_log_ret)).mean(axis=1)
    led: dict[str, Any] = dict(cost_bps=cost_bps, execution=execution, cash_rate=cash_rate,
               borrow_spread_bps=borrow_spread_bps)

    weights = build_weights(detection, log_ret, enter=enter, exit_=exit_,
                            target_vol=target_vol, min_duration=min_duration)
    ledger = run_ledger(weights, simple_ret, **led)

    bh_weights = pd.Series(1.0, index=weights.index, name="weight")
    benchmark = run_ledger(bh_weights, simple_ret, **led)
    ma_w = ma_timing_weights(prices).reindex(weights.index).fillna(0.0)
    ma_rule = run_ledger(ma_w, simple_ret, **led)

    base = build_weights(detection, log_ret, enter=enter, exit_=exit_,
                         target_vol=None, min_duration=min_duration)
    shifted = random_timing_weights(base, seed=random_seed)
    if target_vol is not None:
        scale = vol_target_scale(log_ret, target_vol=target_vol).reindex(weights.index)
        shifted = (shifted * scale).clip(upper=1.5)
        vt_w = vol_target_weights(log_ret, weights.index, target_vol=target_vol)
    else:
        vt_w = bh_weights
    if float(shifted.mean()) > 0:
        shifted = shifted * float(weights.mean()) / float(shifted.mean())
    random_timing = run_ledger(shifted, simple_ret, **led)
    vol_target = run_ledger(vt_w, simple_ret, **led)

    return BacktestResult(
        ledger=ledger,
        benchmark=benchmark,
        ma_rule=ma_rule,
        detection=detection,
        features=features if features is not None else pd.DataFrame(),
        random_timing=random_timing,
        vol_target=vol_target,
    )


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
    min_duration: int = 1,
    execution: ExecutionConfig | None = None,
    cash_rate: float = 0.0,
    borrow_spread_bps: float = 50.0,
) -> BacktestResult:
    """End-to-end walk-forward backtest on a price panel.

    Pipeline: point-in-time features, expanding-window HMM refits every
    ``refit_every`` days, online filtered probabilities, hysteresis and
    vol-targeted weights (optionally with a causal minimum-duration
    confirmation), then the ledger under ``execution`` net of costs and
    financing. Buy-and-hold, the 200d-MA rule, random timing at equal
    exposure and plain vol targeting run over the same dates with the same
    execution and cost model.

    Parameters
    ----------
    prices : (T x N) daily price panel.
    n_states, min_train, refit_every, seed, n_pca : detection settings.
    cost_bps : one-way cost in bp for every ledger.
    enter, exit_ : hysteresis band on ``p_bear``.
    target_vol : annualised vol target (None disables vol targeting).
    feature_kwargs, detect_kwargs : overrides for :func:`build_features` and
        :func:`expanding_fit_detect`.
    min_duration : a regime change reaches the weights only after the new
        regime has held for this many consecutive days (1 = off).
    execution, cash_rate, borrow_spread_bps : see :func:`run_ledger`.
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
    return backtest_from_detection(
        prices, detection, features, cost_bps=cost_bps, enter=enter, exit_=exit_,
        target_vol=target_vol, min_duration=min_duration, execution=execution,
        cash_rate=cash_rate, borrow_spread_bps=borrow_spread_bps, random_seed=seed,
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
