"""Signal construction: combination, IC, decay, deciles, banded rebalancing.

All functions keep the PIT discipline of :mod:`eq_algo.features`: a signal at
date ``t`` uses only information available at ``t``.  Forward returns are the
*targets* — they intentionally look ahead and must never be fed back into a
signal (the backtester never sees them).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .features import cs_zscore

__all__ = [
    "forward_returns",
    "information_coefficient",
    "combine_equal_weight",
    "combine_ic_weighted",
    "signal_decay",
    "decile_portfolios",
    "freeze_signal",
    "apply_rebalance_band",
    "turnover",
]


def forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """``h``-day forward simple return aligned at the *signal* date ``t``.

    ``fwd_t = P_{t+h} / P_t - 1``.  This is a target/label — it uses future
    prices by definition and exists only for evaluation, never as an input.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return prices.shift(-horizon) / prices - 1.0


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame,
                            min_names: int = 3) -> pd.Series:
    """Per-date cross-sectional Spearman rank IC between signal and target.

    For each date, both rows are ranked (average ranks on ties, matching
    ``scipy.stats.spearmanr``) over the names where both are valid, and the
    Pearson correlation of the ranks is returned.  Dates with fewer than
    ``min_names`` valid pairs are NaN.
    """
    signal, fwd = signal.align(fwd, join="inner")
    valid = signal.notna() & fwd.notna()
    s = signal.where(valid)
    f = fwd.where(valid)
    rs = s.rank(axis=1)
    rf = f.rank(axis=1)
    n = valid.sum(axis=1)
    rs_d = rs.sub(rs.mean(axis=1), axis=0)
    rf_d = rf.sub(rf.mean(axis=1), axis=0)
    num = (rs_d * rf_d).sum(axis=1)
    den = np.sqrt((rs_d**2).sum(axis=1) * (rf_d**2).sum(axis=1))
    ic = num / den.mask(den == 0.0)
    ic[n < min_names] = np.nan
    ic.name = "ic"
    return ic


def combine_equal_weight(features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight combination of cross-sectionally z-scored features.

    Each feature is z-scored per date, then the per-name average of the
    available z-scores is taken (missing features are ignored name-by-name).
    """
    if len(features) == 0:
        raise ValueError("features mapping is empty")
    zs = [cs_zscore(f) for f in features.values()]
    total = zs[0].fillna(0.0) * 0.0
    count = zs[0].notna().astype(float) * 0.0
    for z in zs:
        total = total.add(z.fillna(0.0), fill_value=0.0)
        count = count.add(z.notna().astype(float), fill_value=0.0)
    return total / count.mask(count == 0.0)


def combine_ic_weighted(features: Mapping[str, pd.DataFrame], fwd: pd.DataFrame,
                        min_history: int = 60, lag: int = 1) -> pd.DataFrame:
    """IC-weighted combination with point-in-time weights.

    For each feature, the expanding mean of its daily rank IC is computed and
    **lagged by** ``lag + `` the target horizon proxy (the IC at ``t`` uses the
    forward return that resolves at ``t+1``; lagging by ``lag`` days ensures
    the weight applied at ``t`` uses ICs fully observable at ``t``).  Weights
    are floored at zero and normalised per date; if all weights are zero (or
    there is insufficient history) the combination falls back to equal
    weights.

    Parameters
    ----------
    features : mapping name -> DataFrame
    fwd : DataFrame
        1-day forward returns used to score the features.
    min_history : int
        Minimum number of IC observations before IC weights kick in.
    lag : int
        Extra lag (days) applied on top of the 1-day target horizon.
    """
    if len(features) == 0:
        raise ValueError("features mapping is empty")
    names = list(features.keys())
    zs = {k: cs_zscore(features[k]) for k in names}
    index = zs[names[0]].index

    weights = pd.DataFrame(index=index, columns=names, dtype=float)
    for k in names:
        ic = information_coefficient(zs[k], fwd)
        # IC at t is knowable at t+1 (1-day target) -> shift by 1 + lag.
        w = ic.expanding(min_periods=min_history).mean().shift(1 + lag)
        weights[k] = w.reindex(index)
    weights = weights.clip(lower=0.0)
    wsum = weights.sum(axis=1)
    no_info = ~(wsum > 0)
    weights = weights.div(wsum.mask(wsum == 0.0), axis=0)
    weights.loc[no_info, :] = 1.0 / len(names)

    total = None
    wtot = None
    for k in names:
        z = zs[k]
        wk = z.notna().astype(float).mul(weights[k], axis=0)
        contrib = z.fillna(0.0).mul(weights[k], axis=0)
        total = contrib if total is None else total + contrib
        wtot = wk if wtot is None else wtot + wk
    return total / wtot.mask(wtot == 0.0)


def signal_decay(signal: pd.DataFrame, prices: pd.DataFrame,
                 horizons: Sequence[int] = tuple(range(1, 21))) -> pd.DataFrame:
    """Mean IC of the signal against ``h``-day forward returns for each horizon.

    Returns a DataFrame indexed by horizon with columns ``mean_ic``,
    ``ic_std``, ``n_obs`` and ``tstat`` (naive i.i.d. t-stat; use Newey-West
    from :mod:`eq_algo.evaluation` for overlapping horizons > 1, where the
    naive t-stat overstates significance).
    """
    rows = []
    for h in horizons:
        ic = information_coefficient(signal, forward_returns(prices, h)).dropna()
        n = len(ic)
        mean = ic.mean() if n else np.nan
        std = ic.std(ddof=1) if n > 1 else np.nan
        t = mean / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan
        rows.append({"horizon": h, "mean_ic": mean, "ic_std": std, "n_obs": n, "tstat": t})
    return pd.DataFrame(rows).set_index("horizon")


def decile_portfolios(signal: pd.DataFrame, fwd: pd.DataFrame,
                      n_quantiles: int = 10, min_names: int | None = None) -> pd.DataFrame:
    """Mean forward return by signal quantile, per date, plus long-short.

    Names are bucketed per date by signal rank (ties broken by column order
    via ``method='first'``) into ``n_quantiles`` equal-count buckets.
    Column ``Q1`` is the *lowest* signal, ``Q<n>`` the highest, and ``LS``
    is ``Q<n> - Q1`` (equal-weighted within buckets).
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")
    if min_names is None:
        min_names = n_quantiles
    signal, fwd = signal.align(fwd, join="inner")
    valid = signal.notna() & fwd.notna()
    out = {}
    for t in signal.index:
        s = signal.loc[t][valid.loc[t]]
        if len(s) < min_names:
            continue
        r = fwd.loc[t][s.index]
        ranks = s.rank(method="first")
        bucket = np.ceil(ranks / len(s) * n_quantiles).astype(int)
        means = r.groupby(bucket).mean()
        row = {f"Q{q}": means.get(q, np.nan) for q in range(1, n_quantiles + 1)}
        row["LS"] = row[f"Q{n_quantiles}"] - row["Q1"]
        out[t] = row
    cols = [f"Q{q}" for q in range(1, n_quantiles + 1)] + ["LS"]
    return pd.DataFrame.from_dict(out, orient="index").reindex(columns=cols)


def freeze_signal(signal: pd.DataFrame, band: float) -> pd.DataFrame:
    """Turnover control in signal space: refresh a name's signal only when it
    moved more than ``band`` since its last refresh.

    Portfolio construction downstream then sees a *frozen* signal that
    changes rarely, so decile membership — and hence turnover — churns less
    (``trade only when the signal moves > band``).  ``band=0`` returns the
    signal unchanged.  Names with a missing signal today output NaN (they
    leave the tradable universe) but keep their frozen value for when they
    return.

    Parameters
    ----------
    signal : DataFrame
        Signal in comparable units (typically cross-sectional z-scores).
    band : float
        Refresh threshold in signal units; must be >= 0.
    """
    if band < 0:
        raise ValueError("band must be >= 0")
    if band == 0:
        return signal.copy()
    sig = signal.to_numpy(dtype=float)
    out = np.empty_like(sig)
    frozen = np.full(sig.shape[1], np.nan)
    for i in range(sig.shape[0]):
        cur = sig[i]
        fin = np.isfinite(cur)
        upd = fin & (np.isnan(frozen) | (np.abs(cur - frozen) > band))
        frozen = np.where(upd, cur, frozen)
        out[i] = np.where(fin, frozen, np.nan)
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


def apply_rebalance_band(target: pd.DataFrame, band: float) -> pd.DataFrame:
    """Turnover-controlled rebalancing: only trade when the target moved enough.

    For each name, the held weight follows the target only when
    ``|target_t - held_{t-1}| > band``; otherwise the previous weight is kept.
    ``band=0`` reproduces the naive daily rebalance.  NaN targets are treated
    as 0 (name leaves the tradable universe -> position closed if the move
    exceeds the band, mirroring a real no-trade band).

    Parameters
    ----------
    target : DataFrame
        Desired weights, dates x tickers.
    band : float
        No-trade band in absolute weight terms; must be >= 0.
    """
    if band < 0:
        raise ValueError("band must be >= 0")
    tgt = target.fillna(0.0).to_numpy(dtype=float)
    held = np.zeros_like(tgt)
    prev = np.zeros(tgt.shape[1])
    for i in range(tgt.shape[0]):
        move = np.abs(tgt[i] - prev) > band
        prev = np.where(move, tgt[i], prev)
        held[i] = prev
    return pd.DataFrame(held, index=target.index, columns=target.columns)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way turnover per date: ``sum_i |w_{i,t} - w_{i,t-1}|`` (first date
    trades from flat).  NaN weights are treated as 0."""
    w = weights.fillna(0.0)
    d = w.diff()
    d.iloc[0] = w.iloc[0]
    out = d.abs().sum(axis=1)
    out.name = "turnover"
    return out
