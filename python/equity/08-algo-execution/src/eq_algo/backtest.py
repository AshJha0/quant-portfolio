"""Daily cross-sectional long-short backtester with transaction costs.

Timing convention (the single most important line in this module):

    weights decided at close of day t (from a signal using data <= t)
    earn the return from close t to close t+1;
    the cost of trading into those weights is booked on day t itself
    (the spread/impact is paid when the trade is done).

so ``net_t = gross_t - cost_t`` with ``gross_t = sum_i w_{i,t-1} * r_{i,t}``
and ``cost_t`` the cost of moving from ``w_{t-1}`` to ``w_t``.  There is no
path by which a return at ``t+1`` can influence the weight at ``t`` —
enforced by the mutation (no-lookahead) tests.

Cost model (fraction of NAV, per rebalance):

    cost_t = sum_i |dw_i| * ( linear_bps * 1e-4
                              + impact_coef * sigma_i * sqrt(AUM * |dw_i| / ADV$_i) )

i.e. a linear bid/ask + fees term and a square-root market-impact term
(empirical square-root law; see docs/METHODOLOGY.md).  With ``aum=None`` the
impact term is off and the backtest is AUM-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals import freeze_signal

__all__ = ["BacktestConfig", "BacktestResult", "long_short_weights", "run_backtest"]


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for the long-short backtest.

    Parameters
    ----------
    n_quantiles : int
        Decile count for portfolio construction (top vs bottom bucket).
    gross_exposure : float
        Target gross (|long| + |short|) as a fraction of NAV, e.g. 2.0 for a
        1x/1x long-short book.
    max_weight : float
        Per-name absolute weight cap.  If the cap binds, gross falls below
        target rather than concentrating the book (documented behaviour).
    linear_cost_bps : float
        Linear cost per unit turnover, in basis points.
    impact_coef : float
        Square-root impact coefficient ``k`` in ``k * sigma * sqrt(Q/ADV)``.
    aum : float | None
        Assets under management in currency units; ``None`` disables impact.
    rebalance_band : float
        No-trade band in *signal* units (see :func:`eq_algo.signals.freeze_signal`):
        a name's signal is refreshed only when it moved more than the band
        since its last refresh, so decile membership churns less.  For a
        z-scored signal, 0.1-0.5 are typical; 0 = refresh daily.
    """

    n_quantiles: int = 10
    gross_exposure: float = 2.0
    max_weight: float = 0.05
    linear_cost_bps: float = 5.0
    impact_coef: float = 0.1
    aum: float | None = None
    rebalance_band: float = 0.0

    def __post_init__(self) -> None:
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be > 0")
        if self.max_weight <= 0:
            raise ValueError("max_weight must be > 0")
        if self.linear_cost_bps < 0 or self.impact_coef < 0:
            raise ValueError("costs must be >= 0")
        if self.aum is not None and self.aum <= 0:
            raise ValueError("aum must be > 0 when provided")
        if self.rebalance_band < 0:
            raise ValueError("rebalance_band must be >= 0")


@dataclass
class BacktestResult:
    """Backtest output: per-date ledger and the weight matrix actually held."""

    ledger: pd.DataFrame
    weights: pd.DataFrame
    config: BacktestConfig = field(repr=False)

    @property
    def net_returns(self) -> pd.Series:
        return self.ledger["net_ret"]

    @property
    def gross_returns(self) -> pd.Series:
        return self.ledger["gross_ret"]


def long_short_weights(signal: pd.DataFrame, n_quantiles: int = 10,
                       gross_exposure: float = 2.0,
                       max_weight: float = 0.05) -> pd.DataFrame:
    """Dollar-neutral decile long-short weights from a signal.

    Per date: top bucket long, bottom bucket short, equal weight within each
    side, each side scaled to ``gross_exposure / 2``, then per-name weights
    clipped at ``max_weight``; finally both sides are rescaled to the smaller
    side's gross so the book stays exactly dollar neutral (net = 0).  Dates
    with fewer than ``2 * n_quantiles`` valid names produce a flat row.
    """
    idx, cols = signal.index, signal.columns
    w = np.zeros((len(idx), len(cols)))
    sig = signal.to_numpy(dtype=float)
    for i in range(len(idx)):
        row = sig[i]
        valid = np.isfinite(row)
        n = int(valid.sum())
        if n < 2 * n_quantiles:
            continue
        vals = row[valid]
        order = np.argsort(np.argsort(vals, kind="stable"), kind="stable")  # 0..n-1 ranks
        bucket = np.ceil((order + 1) / n * n_quantiles).astype(int)
        longs = bucket == n_quantiles
        shorts = bucket == 1
        side = gross_exposure / 2.0
        wl = np.zeros(n)
        wl[longs] = side / longs.sum()
        wl[shorts] = -side / shorts.sum()
        wl = np.clip(wl, -max_weight, max_weight)
        long_g = wl[wl > 0].sum()
        short_g = -wl[wl < 0].sum()
        tgt = min(long_g, short_g)
        if tgt > 0:
            if long_g > 0:
                wl[wl > 0] *= tgt / long_g
            if short_g > 0:
                wl[wl < 0] *= tgt / short_g
        w[i, valid] = wl
    return pd.DataFrame(w, index=idx, columns=cols)


def _rebalance_cost(dw: np.ndarray, sigma: np.ndarray, adv_dollars: np.ndarray,
                    cfg: BacktestConfig) -> float:
    """Cost (fraction of NAV) of a weight change vector ``dw``."""
    lin = cfg.linear_cost_bps * 1e-4 * np.abs(dw).sum()
    if cfg.aum is None or cfg.impact_coef == 0.0:
        return float(lin)
    absw = np.abs(dw)
    with np.errstate(divide="ignore", invalid="ignore"):
        part = np.where(absw > 0, cfg.aum * absw / adv_dollars, 0.0)
    impact = cfg.impact_coef * sigma * np.sqrt(part) * absw
    return float(lin + np.nansum(impact))


def run_backtest(prices: pd.DataFrame, signal: pd.DataFrame,
                 config: BacktestConfig | None = None,
                 volumes: pd.DataFrame | None = None,
                 vol_window: int = 63, adv_window: int = 21) -> BacktestResult:
    """Run the daily long-short backtest.

    Parameters
    ----------
    prices : DataFrame
        Close prices, dates x tickers.
    signal : DataFrame
        PIT signal aligned to ``prices`` (value at t uses data <= t).
    config : BacktestConfig
    volumes : DataFrame, optional
        Share volumes; required if ``config.aum`` is set (impact needs ADV$).
        ADV$ and daily vol are estimated with trailing windows (PIT).
    vol_window, adv_window : int
        Trailing windows (days) for the per-name daily-vol and ADV$ estimates
        used by the impact model.

    Returns
    -------
    BacktestResult
        Ledger columns: ``gross_ret, cost, net_ret, turnover, gross_exposure,
        net_exposure, n_long, n_short``.  Row at date ``t``: ``gross_ret`` is
        the return earned from ``t-1`` close to ``t`` close on the weights
        held over that interval; ``turnover``/``cost`` describe the trade
        done at the close of ``t``; ``net_ret = gross_ret - cost``.
    """
    cfg = config or BacktestConfig()
    signal = signal.reindex(index=prices.index, columns=prices.columns)
    if cfg.aum is not None and volumes is None:
        raise ValueError("volumes are required when aum is set (impact model needs ADV$)")

    tradable = prices.notna()
    sig = signal.where(tradable)
    sig = freeze_signal(sig, cfg.rebalance_band)
    held = long_short_weights(sig, cfg.n_quantiles, cfg.gross_exposure, cfg.max_weight)

    rets = prices.pct_change()
    n_dates = len(prices.index)
    sigma_np = adv_np = None
    if cfg.aum is not None:
        daily_vol = prices.pct_change().rolling(vol_window, min_periods=vol_window // 2).std(ddof=1)
        adv_dollars = (prices * volumes).rolling(adv_window, min_periods=1).mean()
        sigma_np = daily_vol.to_numpy(dtype=float)
        adv_np = adv_dollars.to_numpy(dtype=float)

    w_np = held.to_numpy(dtype=float)
    r_np = rets.to_numpy(dtype=float)
    rows = []
    prev_w = np.zeros(w_np.shape[1])
    for i in range(n_dates):
        r = np.nan_to_num(r_np[i], nan=0.0)
        gross_ret = float(prev_w @ r)
        w_now = w_np[i]
        dw = w_now - prev_w
        to = float(np.abs(dw).sum())
        if cfg.aum is not None:
            sig_i = np.nan_to_num(sigma_np[i], nan=0.0)
            adv_i = np.where(np.isfinite(adv_np[i]) & (adv_np[i] > 0), adv_np[i], np.inf)
            cost = _rebalance_cost(dw, sig_i, adv_i, cfg)
        else:
            cost = _rebalance_cost(dw, np.zeros_like(dw), np.ones_like(dw), cfg)
        rows.append({
            "gross_ret": gross_ret,
            "cost": cost,
            "net_ret": gross_ret - cost,
            "turnover": to,
            "gross_exposure": float(np.abs(w_now).sum()),
            "net_exposure": float(w_now.sum()),
            "n_long": int((w_now > 0).sum()),
            "n_short": int((w_now < 0).sum()),
        })
        prev_w = w_now
    ledger = pd.DataFrame(rows, index=prices.index)
    return BacktestResult(ledger=ledger, weights=held, config=cfg)
