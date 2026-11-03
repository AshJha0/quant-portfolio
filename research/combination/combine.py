"""Carry + value + momentum combination with vol targeting and limits (W23 part 2).

Plain idea: three currency signals (carry, value, momentum) are each noisy.
Ranked the same way and averaged, their errors partly cancel, so the blend
should be steadier than any one of them (Asness, Moskowitz and Pedersen
2013, section V; Kroencke, Schindler and Schrimpf 2014). Who pays: carry is
paid by investors hedging crash risk in high-rate currencies (Lustig,
Roussanov and Verdelhan 2011); value by slow mean reversion of real exchange
rates (Menkhoff, Sarno, Schmeling and Schrimpf 2017); momentum by slow
diffusion of information (Menkhoff et al. 2012). These are known premia, not
new alpha, and every result here is labelled that way.

Formal statement. At each month end ``m``:

1. ``z_k = (s_k - mean(s_k)) / sd(s_k)`` across live currencies, per signal
   ``k``, clipped to [-3, 3];
2. ``c = mean_k z_k`` over the signals present for that currency (a currency
   needs at least ``min_signals``);
3. raw weights ``w = c / sum|c|`` (dollar-neutral, unit gross);
4. ex-ante covariance ``S`` is an EWMA of daily or monthly excess returns
   (half-life ``h``) shrunk to its diagonal: ``(1 - d) S + d diag(S)``;
5. the weights are scaled so ``sqrt(w' S w * periods) = target_vol``, capped
   at ``max_gross``;
6. per-currency limit ``|w_i| <= max_weight``, applied after scaling by
   clipping (the clip only lowers risk);
7. optional cost-aware sizing: ``fx_port.cost_qp.rebalance`` with ``mu``
   proportional to ``c``, the shrunk ``S`` and tiered costs, no budget.

Returns: weights set at month end ``m`` earn the excess return of month
``m+1``; cost is ``sum_i tier_bp_i |w_i(m) - w_i(m-1)|`` in bp.

Worked example (``tests/test_combine.py``): three currencies with carry
z-scores (-1, 0, 1), value (1, 0, -1) and momentum (-1, 0, 1) give a
combination of (-1/3, 0, 1/3), raw weights (-0.5, 0, 0.5); with an identity
monthly covariance of 0.01 and target 10 % annual the weights scale by
``0.1 / sqrt(0.5 * 0.01 * 12) = 0.4082``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CombinationSpec:
    """Every knob of the combination, fixed in a registration."""

    target_vol: float = 0.10
    periods_per_year: int = 12
    halflife: float = 12.0
    shrink: float = 0.5
    max_gross: float = 3.0
    max_weight: float = 0.25
    min_signals: int = 2
    clip_z: float = 3.0
    min_history: int = 24

    def __post_init__(self) -> None:
        if not 0.0 <= self.shrink <= 1.0:
            raise ValueError("shrink must lie in [0, 1]")
        if self.target_vol <= 0 or self.max_gross <= 0 or self.max_weight <= 0:
            raise ValueError("target_vol, max_gross and max_weight must be positive")
        if self.halflife <= 0:
            raise ValueError("halflife must be positive")


def cross_zscore(signal: pd.DataFrame, live: pd.DataFrame | None = None,
                 clip: float = 3.0) -> pd.DataFrame:
    """Row-wise z-score over live, non-missing currencies, clipped."""
    s = signal.astype(float)
    if live is not None:
        s = s.where(live.reindex_like(s).fillna(False).astype(bool))
    mu = s.mean(axis=1)
    sd = s.std(axis=1, ddof=1)
    z = s.sub(mu, axis=0).div(sd.where(sd > 0), axis=0)
    return z.clip(-clip, clip)


def combine_signals(signals: Mapping[str, pd.DataFrame], live: pd.DataFrame | None = None,
                    min_signals: int = 2, clip: float = 3.0) -> pd.DataFrame:
    """Equal-weight mean of the cross-sectional z-scores of each signal."""
    if not signals:
        raise ValueError("no signals")
    zs = [cross_zscore(s, live, clip) for s in signals.values()]
    idx = zs[0].index
    cols = zs[0].columns
    for z in zs[1:]:
        idx = idx.union(z.index)
        cols = cols.union(z.columns)
    stack = np.stack([z.reindex(index=idx, columns=cols).to_numpy(float) for z in zs])
    n = np.sum(~np.isnan(stack), axis=0)
    mean = _nanmean(stack)
    mean[n < min_signals] = np.nan
    return pd.DataFrame(mean, index=idx, columns=cols)


def _nanmean(stack: np.ndarray) -> np.ndarray:
    tot = np.nansum(stack, axis=0)
    n = np.sum(~np.isnan(stack), axis=0)
    out = np.full(tot.shape, np.nan)
    np.divide(tot, n, out=out, where=n > 0)
    return out


def raw_weights(combo: pd.Series) -> pd.Series:
    """Demeaned, unit-gross weights from one row of combined scores."""
    c = combo.dropna()
    if len(c) < 2:
        return pd.Series(0.0, index=combo.index)
    c = c - c.mean()
    g = c.abs().sum()
    w = c / g if g > 0 else c * 0.0
    return w.reindex(combo.index).fillna(0.0)


def ewma_cov(returns: pd.DataFrame, halflife: float) -> np.ndarray:
    """EWMA covariance (zero-mean, as RiskMetrics 1996), missing returns as 0.

    Weights ``lambda^(T-1-t)`` with ``lambda = 0.5 ** (1/halflife)``,
    normalised to sum to one.
    """
    x = returns.fillna(0.0).to_numpy(float)
    t = x.shape[0]
    if t == 0:
        raise ValueError("no returns")
    lam = 0.5 ** (1.0 / halflife)
    wts = lam ** np.arange(t - 1, -1, -1, dtype=float)
    wts /= wts.sum()
    return (x * wts[:, None]).T @ x


def shrink_to_diag(cov: np.ndarray, d: float) -> np.ndarray:
    return (1.0 - d) * cov + d * np.diag(np.diag(cov))


def vol_target(w: np.ndarray, cov: np.ndarray, spec: CombinationSpec) -> np.ndarray:
    """Scale to the target annual vol, cap gross, then clip per-name limits."""
    var = float(w @ cov @ w) * spec.periods_per_year
    if not var > 0:
        return np.zeros_like(w)
    k = spec.target_vol / np.sqrt(var)
    g = np.abs(w).sum()
    if g * k > spec.max_gross:
        k = spec.max_gross / g
    return np.clip(w * k, -spec.max_weight, spec.max_weight)


Sizer = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def cost_qp_sizer(risk_aversion: float = 10.0, alpha_scale: float = 0.01) -> Sizer:
    """Sizing through ``fx_port.cost_qp.rebalance``: ``mu = alpha_scale * w_target``
    direction, tiered costs in fractions, no budget (the book is self-financing)."""
    from fx_port.cost_qp import rebalance

    def size(w_target: np.ndarray, cov: np.ndarray, w_prev: np.ndarray,
             cost_frac: np.ndarray) -> np.ndarray:
        mu = alpha_scale * w_target
        r = rebalance(mu, cov, w_prev, risk_aversion, cost=cost_frac, budget=False)
        return np.asarray(r.weights, dtype=float)

    return size


@dataclass(frozen=True)
class CombinationResult:
    weights: pd.DataFrame      # month x currency, set at month end
    gross: pd.Series           # next-month gross return of those weights
    cost: pd.Series            # cost charged at the rebalance, as a return
    net: pd.Series
    turnover: pd.Series        # sum |dw| at each rebalance


def run(signals: Mapping[str, pd.DataFrame], next_ret: pd.DataFrame,
        cost_bp: Mapping[str, float], spec: CombinationSpec | None = None,
        live: pd.DataFrame | None = None, sizer: Sizer | None = None) -> CombinationResult:
    """Monthly combination backtest.

    ``next_ret`` row ``m`` is the excess return over the month after ``m``
    (the W6 panel convention). The covariance at ``m`` uses rows strictly
    before ``m`` (returns already realised by the month-end ``m``), so there
    is no look-ahead.
    """
    spec = spec or CombinationSpec()
    combo = combine_signals(signals, live, spec.min_signals, spec.clip_z)
    months = combo.index.intersection(next_ret.index)
    cols = list(combo.columns.intersection(next_ret.columns))
    combo = combo.loc[months, cols]
    nr = next_ret.loc[months, cols].astype(float)
    cvec = np.array([cost_bp.get(c, 15.0) for c in cols]) * 1e-4
    w_prev = np.zeros(len(cols))
    rows_w, gross, cost, turn = [], [], [], []
    for i, m in enumerate(months):
        # row j is the return over month j+1, realised by month end j+1 <= m
        # exactly when j < i: every row before m is known at m.
        hist = nr.iloc[:i]
        if len(hist) < spec.min_history:
            w = np.zeros(len(cols))
        else:
            cov = shrink_to_diag(ewma_cov(hist, spec.halflife), spec.shrink)
            w = vol_target(raw_weights(combo.loc[m]).to_numpy(float), cov, spec)
            if sizer is not None:
                w = np.clip(sizer(w, cov, w_prev, cvec), -spec.max_weight, spec.max_weight)
        r = nr.loc[m].to_numpy(float)
        held = np.where(np.isnan(r), 0.0, w)
        dw = np.abs(held - w_prev)
        rows_w.append(held)
        turn.append(float(dw.sum()))
        cost.append(float(dw @ cvec))
        gross.append(float(np.nansum(held * np.nan_to_num(r))))
        w_prev = held
    W = pd.DataFrame(rows_w, index=months, columns=cols)
    g = pd.Series(gross, index=months)
    c = pd.Series(cost, index=months)
    return CombinationResult(W, g, c, g - c, pd.Series(turn, index=months))


def sharpe(x: pd.Series, periods: int = 12) -> float:
    x = x.dropna()
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(periods)) if sd > 0 else float("nan")
