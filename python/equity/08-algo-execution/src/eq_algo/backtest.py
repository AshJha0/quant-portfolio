"""Daily cross-sectional long-short backtester with transaction costs.

Execution convention (shared by every strategy project in this repository):

    Signals use data up to the close of day t. Orders fill at a modelled
    t+1 price, p_t + f (p_{t+1} - p_t) with f = 0.5 by default; a
    close-auction variant fills at p_t and pays an auction cost.

In return terms, with ``r = p_{t+1}/p_t - 1`` and the default mode
``"next"``, the weights held before the trade earn ``f r`` (close t to the
fill) and the new weights earn ``(1 + r)/(1 + f r) - 1`` (fill to close
t+1). The trade and its cost are booked in the ledger row of day t+1, the
day it fills. ``fill_fraction = 1`` fills at the next close. In mode
``"close_auction"`` the new weights earn the whole of ``r`` and every unit of
turnover pays ``auction_cost_bp`` on top of the normal cost.

Cost model (fraction of NAV, per rebalance):

    cost = sum_i |dw_i| * ( linear_bps * 1e-4
                            + impact_coef * sigma_i * sqrt(AUM * |dw_i| / ADV$_i) )

a linear spread and fee term plus a square-root market-impact term. The
default ``impact_coef = 0.7`` sits in the middle of the 0.5-1 range of the
empirical coefficient Y (Almgren et al. 2005; Toth et al. 2011). ``sigma_i``
and ``ADV$_i`` are trailing estimates known at the decision close. The short
book pays a borrow fee ``borrow_bps_annual / 252`` per day on its gross.
With ``aum=None`` the impact term is off and the backtest is AUM-independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from qp_core.execution import ExecutionConfig
from qp_core.stats import Financing, warn_no_financing

from .signals import freeze_signal

__all__ = ["BacktestConfig", "BacktestResult", "ExecutionConfig", "capacity_backtest", "long_short_weights", "run_backtest"]





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
        Square-root impact coefficient ``Y`` in ``Y * sigma * sqrt(Q/ADV)``.
        Default 0.7, inside the empirical 0.5-1 range.
    borrow_bps_annual : float
        Annual borrow fee on the gross short book, in bp (charged daily /252).
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
    impact_coef: float = 0.7
    aum: float | None = None
    rebalance_band: float = 0.0
    borrow_bps_annual: float = 0.0

    def __post_init__(self) -> None:
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if self.gross_exposure <= 0:
            raise ValueError("gross_exposure must be > 0")
        if self.max_weight <= 0:
            raise ValueError("max_weight must be > 0")
        for name in ("gross_exposure", "max_weight", "linear_cost_bps", "impact_coef",
                     "rebalance_band", "borrow_bps_annual"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.linear_cost_bps < 0 or self.impact_coef < 0 or self.borrow_bps_annual < 0:
            raise ValueError("costs must be >= 0")
        if self.aum is not None and not np.isfinite(float(self.aum)):
            raise ValueError("aum must be finite")
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
    def excess_returns(self) -> pd.Series:
        """Net return minus the risk-free return; use these for Sharpe (W6d)."""
        return self.ledger["excess_ret"]

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
                 vol_window: int = 63, adv_window: int = 21,
                 execution: ExecutionConfig | None = None,
                 financing: Financing | None = None) -> BacktestResult:
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
        ADV$ and daily vol are trailing estimates up to and including t.
    vol_window, adv_window : int
        Trailing windows (days) for the per-name daily-vol and ADV$ estimates
        used by the impact model.
    execution : ExecutionConfig, optional
        Fill convention; default ``ExecutionConfig()`` (fill at
        ``p_t + 0.5 (p_{t+1} - p_t)``).
    financing : qp_core.stats.Financing, optional
        Rate and spreads for cash (W6d). The cash residual
        ``1 - net_exposure`` of the book held into day t earns, or when
        negative pays, the rate over the period: ledger column ``financing``,
        included in ``net_ret``. ``risk_free`` and ``excess_ret`` are added.
        ``None`` keeps zero-rate cash and emits ``FinancingWarning``.

    Returns
    -------
    BacktestResult
        ``weights`` row t is the target decided at the close of t. Ledger row
        t: ``gross_ret`` is the return earned over day t (close t-1 to
        close t) including the split around the fill; ``turnover``, ``cost``
        (linear + impact + auction) and ``borrow`` describe the trade filled
        on day t and the short book held after it;
        ``net_ret = gross_ret - cost - borrow + financing``;
        ``excess_ret = net_ret - risk_free``.
    """
    cfg = config or BacktestConfig()
    ex = execution or ExecutionConfig()
    signal = signal.reindex(index=prices.index, columns=prices.columns)
    if cfg.aum is not None and volumes is None:
        raise ValueError("volumes are required when aum is set (impact model needs ADV$)")
    p_np = prices.to_numpy(dtype=float)
    if np.isinf(p_np).any() or (np.nan_to_num(p_np, nan=1.0) <= 0).any():
        raise ValueError("prices must be positive and finite (NaN marks untradable)")

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
    n_names = w_np.shape[1]
    f = ex.fill_fraction
    auction = ex.mode == "close_auction"
    borrow_daily = cfg.borrow_bps_annual * 1e-4 / 252.0
    if financing is None:
        warn_no_financing("eq_algo.run_backtest")
        rf_all = np.zeros(n_dates)
        tau_all = np.zeros(n_dates)
    else:
        rf_all, tau_all = financing.accruals(prices.index)

    def trade_cost(i_dec: int, dw: np.ndarray) -> float:
        if cfg.aum is not None:
            sig_np, a_np = cast(np.ndarray, sigma_np), cast(np.ndarray, adv_np)
            sig_i = np.nan_to_num(sig_np[i_dec], nan=0.0)
            adv_i = np.where(np.isfinite(a_np[i_dec]) & (a_np[i_dec] > 0),
                             a_np[i_dec], np.inf)
            c = _rebalance_cost(dw, sig_i, adv_i, cfg)
        else:
            c = _rebalance_cost(dw, np.zeros_like(dw), np.ones_like(dw), cfg)
        if auction:
            c += ex.auction_cost_bp * 1e-4 * float(np.abs(dw).sum())
        return c

    rows = []
    held_w = np.zeros(n_names)      # weights held into day i
    for i in range(n_dates):
        r = np.nan_to_num(r_np[i], nan=0.0)
        if auction:
            # the target decided at close i fills at close i
            gross_ret = float(held_w @ r)
            w_new = w_np[i]
            dw = w_new - held_w
            cost = trade_cost(i, dw)
        else:
            # the target decided at close i-1 fills during day i
            w_new = w_np[i - 1] if i > 0 else np.zeros(n_names)
            dw = w_new - held_w
            r_pre = f * r
            r_post = (1.0 + r) / (1.0 + f * r) - 1.0
            gross_ret = float(held_w @ r_pre + w_new @ r_post)
            cost = trade_cost(i - 1, dw) if i > 0 else 0.0
        borrow = borrow_daily * float(-w_new[w_new < 0].sum())
        fin = (financing.cash_return(1.0 - float(held_w.sum()), rf_all[i], tau_all[i])
               if financing is not None else 0.0)
        rows.append({
            "gross_ret": gross_ret,
            "cost": cost,
            "borrow": borrow,
            "financing": fin,
            "net_ret": gross_ret - cost - borrow + fin,
            "risk_free": float(rf_all[i]),
            "excess_ret": gross_ret - cost - borrow + fin - float(rf_all[i]),
            "turnover": float(np.abs(dw).sum()),
            "gross_exposure": float(np.abs(w_new).sum()),
            "net_exposure": float(w_new.sum()),
            "n_long": int((w_new > 0).sum()),
            "n_short": int((w_new < 0).sum()),
        })
        held_w = w_new
    ledger = pd.DataFrame(rows, index=prices.index)
    return BacktestResult(ledger=ledger, weights=held, config=cfg)


def capacity_backtest(prices: pd.DataFrame, signal: pd.DataFrame, volumes: pd.DataFrame,
                      config: BacktestConfig, aum_grid, impact_coefs=(0.3, 0.7, 1.0),
                      execution: ExecutionConfig | None = None, start: int = 0,
                      adv_window: int = 21,
                      financing: Financing | None = None) -> pd.DataFrame:
    """Capacity by re-running the full backtest at each AUM and impact coefficient.

    Every run uses per-name trailing ADV$ and daily vol known at the decision
    close, so the cost of each trade reflects that name's liquidity on that
    day. Rows are ``(impact_coef, aum)``; columns: ``ann_net_return``,
    ``net_sharpe``, ``drag_bps_per_day`` (mean cost plus borrow),
    ``p95_participation`` (95th percentile of traded dollars over trailing
    ADV$ across all name-day trades). ``start`` drops warm-up rows.

    ``net_sharpe`` is computed on excess returns (``excess_ret``: net return
    less the risk-free return). ``financing`` is passed to ``run_backtest``;
    with ``financing=None`` the excess and net returns are identical.
    """
    if volumes is None:
        raise ValueError("volumes are required for a capacity run")
    if start < 0 or start >= len(prices.index) - 2:
        raise ValueError("start must leave at least two rows")
    from dataclasses import replace as _replace

    adv = (prices * volumes).rolling(adv_window, min_periods=1).mean().to_numpy(dtype=float)
    rows = []
    for y in impact_coefs:
        if not np.isfinite(y) or y < 0:
            raise ValueError("impact coefficients must be finite and >= 0")
        for aum in aum_grid:
            if not np.isfinite(aum) or aum <= 0:
                raise ValueError("AUM values must be finite and > 0")
            cfg = _replace(config, impact_coef=float(y), aum=float(aum))
            res = run_backtest(prices, signal, cfg, volumes=volumes,
                               adv_window=adv_window, execution=execution,
                               financing=financing)
            led = res.ledger.iloc[start:]
            net = led["net_ret"]
            exc = led["excess_ret"]
            dw = res.weights.diff().to_numpy(dtype=float)[start:]
            a = adv[start:]
            with np.errstate(divide="ignore", invalid="ignore"):
                part = np.abs(dw) * aum / a
            part = part[np.isfinite(part) & (np.abs(dw) > 0)]
            rows.append({
                "impact_coef": float(y),
                "aum": float(aum),
                "ann_net_return": float(net.mean() * 252.0),
                "net_sharpe": float(exc.mean() / exc.std(ddof=1) * np.sqrt(252.0)),
                "drag_bps_per_day": float((led["cost"] + led["borrow"]).mean() * 1e4),
                "p95_participation": float(np.quantile(part, 0.95)) if part.size else 0.0,
            })
    return pd.DataFrame(rows).set_index(["impact_coef", "aum"])
