"""Model validation from scratch: AUC/Gini/KS, Hosmer-Lemeshow, PSI, bootstrap,
and a panel autocorrelation warning utility.

All statistics are implemented from first principles (rank statistics, chi
square, entropy distance); sklearn/scipy are used only as cross-checks in the
test suite (AUC matches ``sklearn.metrics.roc_auc_score`` to 1e-12 including
ties, via the same average-rank Mann-Whitney construction).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "auc",
    "gini",
    "ks_statistic",
    "hosmer_lemeshow",
    "HosmerLemeshowResult",
    "psi",
    "bootstrap_auc_ci",
    "within_country_autocorrelation",
]


def _check_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be binary {0,1}")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("y must contain both classes")
    return y


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic with average ranks (tie-exact).

    ``AUC = (R1 - n1(n1+1)/2) / (n1 n0)`` where ``R1`` is the rank-sum of the
    positive class using average ranks — identical (to floating point) to
    ``sklearn.metrics.roc_auc_score`` including tied scores.
    """
    y = _check_binary(y)
    s = np.asarray(score, dtype=float).ravel()
    if s.shape != y.shape:
        raise ValueError("score and y must have equal length")
    ranks = stats.rankdata(s, method="average")
    n1 = y.sum()
    n0 = len(y) - n1
    r1 = ranks[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def gini(y: np.ndarray, score: np.ndarray) -> float:
    """Gini / accuracy ratio: ``2*AUC - 1``."""
    return 2.0 * auc(y, score) - 1.0


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance between score distributions of the classes.

    ``KS = max_t |F_bad(t) - F_good(t)|`` over the empirical CDFs of the
    score among defaulters and non-defaulters.
    """
    y = _check_binary(y)
    s = np.asarray(score, dtype=float).ravel()
    order = np.argsort(s, kind="mergesort")
    ys = y[order]
    n1 = ys.sum()
    n0 = len(ys) - n1
    cum_bad = np.cumsum(ys) / n1
    cum_good = np.cumsum(1 - ys) / n0
    # evaluate at distinct-score right edges only (ties handled jointly)
    ss = s[order]
    last_of_value = np.r_[ss[1:] != ss[:-1], True]
    return float(np.max(np.abs(cum_bad[last_of_value] - cum_good[last_of_value])))


@dataclass(frozen=True)
class HosmerLemeshowResult:
    """Hosmer-Lemeshow goodness-of-fit test output."""

    chi2: float
    p_value: float
    dof: int
    table: pd.DataFrame  # per-group observed vs expected


def hosmer_lemeshow(
    y: np.ndarray,
    pd_hat: np.ndarray,
    n_groups: int = 10,
) -> HosmerLemeshowResult:
    """Hosmer-Lemeshow calibration test on ``n_groups`` predicted-PD groups.

    Groups are equal-count by sorted predicted PD.  For group g with n_g
    rows, O_g observed defaults and E_g = sum of predicted PDs:

    ``chi2 = sum_g (O_g - E_g)^2 / (E_g (1 - E_g/n_g))``,
    referred to chi-square with ``n_groups - 2`` degrees of freedom.

    Low p-value = evidence of *mis-calibration* (predicted PD levels wrong),
    even when discrimination (AUC) is fine — the typical failure in a
    contagion year.
    """
    y = _check_binary(y)
    p = np.asarray(pd_hat, dtype=float).ravel()
    if np.any((p <= 0) | (p >= 1)):
        p = np.clip(p, 1e-12, 1 - 1e-12)
    if n_groups < 3:
        raise ValueError("n_groups must be >= 3")
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, n_groups)
    rows = []
    chi2 = 0.0
    for g, idx in enumerate(groups):
        n_g = len(idx)
        o = float(y[idx].sum())
        e = float(p[idx].sum())
        pbar = e / n_g
        denom = e * (1.0 - pbar)
        chi2 += (o - e) ** 2 / max(denom, 1e-12)
        rows.append({"group": g + 1, "n": n_g, "observed": o, "expected": e,
                     "mean_pd": pbar, "obs_rate": o / n_g})
    dof = n_groups - 2
    p_value = float(stats.chi2.sf(chi2, dof))
    return HosmerLemeshowResult(float(chi2), p_value, dof, pd.DataFrame(rows))


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    edges: np.ndarray | None = None,
) -> float:
    """Population Stability Index between a baseline and a current sample.

    ``PSI = sum_i (a_i - e_i) * ln(a_i / e_i)`` over shared bins, where
    ``e_i``/``a_i`` are the baseline/current population shares.  Bins default
    to baseline deciles.  Empty cells are floored at 1e-6 share (documented
    convention; keeps PSI finite).  Rules of thumb: <0.10 stable, 0.10-0.25
    monitor, >0.25 shifted.
    """
    e = np.asarray(expected, dtype=float).ravel()
    a = np.asarray(actual, dtype=float).ravel()
    if e.size == 0 or a.size == 0:
        raise ValueError("both samples must be non-empty")
    if edges is None:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    edges = np.asarray(edges, dtype=float)
    inner = edges[1:-1]
    ce = np.bincount(np.searchsorted(inner, e, side="left"), minlength=edges.size - 1)
    ca = np.bincount(np.searchsorted(inner, a, side="left"), minlength=edges.size - 1)
    pe = np.maximum(ce / ce.sum(), 1e-6)
    pa = np.maximum(ca / ca.sum(), 1e-6)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def bootstrap_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int = 500,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for AUC (resampling rows with replacement).

    With a low-default portfolio the CI is *wide* — the honest statement of
    how little the default history pins down discrimination (documented and
    demonstrated in VALIDATION.md).

    Returns
    -------
    (auc_hat, lo, hi)
    """
    y = _check_binary(y)
    s = np.asarray(score, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    point = auc(y, s)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        vals.append(auc(yb, s[idx]))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return point, float(lo), float(hi)


def within_country_autocorrelation(
    df: pd.DataFrame,
    col: str,
    country_col: str = "country",
    warn_threshold: float = 0.10,
) -> pd.Series:
    """Lag-1 autocorrelation of ``col`` within each country; warns if pervasive.

    Use on model residuals (``default - pd_hat``): substantial within-country
    autocorrelation means rows are not independent, standard errors and
    binomial-based tests (HL) are anti-conservative, and a random row split
    would leak.  Emits a ``UserWarning`` when the mean autocorrelation over
    countries exceeds ``warn_threshold``.

    Returns
    -------
    pandas.Series
        Autocorrelation per country (NaN for countries with < 3 usable rows
        or zero variance).
    """
    if col not in df.columns:
        raise ValueError(f"column {col!r} not in frame")
    out = {}
    for c, g in df.sort_values("year").groupby(country_col):
        v = g[col].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if v.size < 3 or np.std(v[:-1]) == 0 or np.std(v[1:]) == 0:
            out[c] = np.nan
            continue
        out[c] = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    ser = pd.Series(out, name=f"lag1_autocorr[{col}]")
    mean_ac = ser.mean(skipna=True)
    if np.isfinite(mean_ac) and mean_ac > warn_threshold:
        warnings.warn(
            f"mean within-country lag-1 autocorrelation of {col!r} is "
            f"{mean_ac:.3f} > {warn_threshold}: panel rows are serially "
            "dependent — use time/country splits, not random row splits, and "
            "treat i.i.d.-based standard errors as anti-conservative.",
            UserWarning,
            stacklevel=2,
        )
    return ser
