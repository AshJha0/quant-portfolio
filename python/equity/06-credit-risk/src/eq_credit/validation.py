"""Model validation: discrimination, calibration, stability, rank ordering.

Discrimination
--------------
* ROC/AUC from scratch (threshold sweep + trapezoid — numerically identical
  to sklearn's ``roc_auc_score``), Gini = 2*AUC - 1, KS statistic (exact,
  over all thresholds) and a score-decile KS table.
* Bootstrap percentile CIs for AUC (seeded).

Calibration
-----------
* Hosmer-Lemeshow decile chi-square (df = groups - 2), reliability
  (calibration) table, Brier score.

Stability
---------
* PSI between a baseline (train/expected) and a comparison (OOT/actual)
  sample: ``PSI = sum (p_act - p_exp) * ln(p_act / p_exp)`` with standard
  thresholds: < 0.10 stable, 0.10-0.25 monitor, > 0.25 shifted (recalibrate).

Conventions: higher score = safer; ``y`` is the default indicator (1 = bad).
AUC is reported for the PD (or -score) direction: AUC > 0.5 means defaults
get higher PDs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "roc_curve_points",
    "roc_auc",
    "gini",
    "ks_statistic",
    "ks_table",
    "bootstrap_auc_ci",
    "brier_score",
    "hosmer_lemeshow",
    "calibration_table",
    "psi_from_proportions",
    "psi",
    "psi_report",
    "psi_status",
    "decile_table",
    "is_monotone",
]


def _check_binary(y: np.ndarray) -> np.ndarray:
    ya = np.asarray(y, dtype=float).ravel()
    if len(ya) == 0:
        raise ValueError("empty target")
    if not np.isin(ya, [0.0, 1.0]).all():
        raise ValueError("y must be binary 0/1")
    if ya.sum() == 0 or ya.sum() == len(ya):
        raise ValueError(
            "target has a single class (zero defaults or zero goods): "
            "discrimination metrics undefined"
        )
    return ya


def roc_curve_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve (FPR, TPR) treating higher ``s`` as more likely default.

    One point per distinct threshold, plus the (0,0) origin.
    """
    ya = _check_binary(y)
    sa = np.asarray(s, dtype=float).ravel()
    order = np.argsort(-sa, kind="mergesort")
    ys = ya[order]
    ss = sa[order]
    distinct = np.where(np.diff(ss))[0]
    idx = np.r_[distinct, len(ss) - 1]
    tps = np.cumsum(ys)[idx]
    fps = (idx + 1) - tps
    tpr = np.r_[0.0, tps / ys.sum()]
    fpr = np.r_[0.0, fps / (len(ys) - ys.sum())]
    return fpr, tpr


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """AUC from scratch: trapezoid over the ROC curve (ties handled exactly
    as in sklearn — matches ``roc_auc_score`` to machine precision)."""
    fpr, tpr = roc_curve_points(y, s)
    return float(np.trapezoid(tpr, fpr))


def gini(y: np.ndarray, s: np.ndarray) -> float:
    """Gini (accuracy ratio) = 2*AUC - 1."""
    return 2.0 * roc_auc(y, s) - 1.0


def ks_statistic(y: np.ndarray, s: np.ndarray) -> float:
    """Exact KS: max distance between bad and good CDFs of ``s``."""
    ya = _check_binary(y)
    sa = np.asarray(s, dtype=float).ravel()
    fpr, tpr = roc_curve_points(ya, sa)
    return float(np.max(np.abs(tpr - fpr)))


def ks_table(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Score-decile KS table (higher score = safer).

    Columns: decile (1 = riskiest, lowest scores), score range, counts, bad
    rate, cumulative bad %, cumulative good %, KS = |cum_bad% - cum_good%|.
    """
    ya = _check_binary(y)
    sa = np.asarray(score, dtype=float).ravel()
    ranks = pd.qcut(pd.Series(sa).rank(method="first"), n_bins, labels=False)
    rows = []
    cum_bad = cum_good = 0.0
    tot_bad, tot_good = ya.sum(), len(ya) - ya.sum()
    for d in range(n_bins):
        m = (ranks == d).to_numpy()
        nb, ng = ya[m].sum(), m.sum() - ya[m].sum()
        cum_bad += nb
        cum_good += ng
        rows.append(
            {
                "decile": d + 1,
                "score_min": sa[m].min(),
                "score_max": sa[m].max(),
                "n": int(m.sum()),
                "n_bad": int(nb),
                "bad_rate": nb / m.sum(),
                "cum_bad_pct": cum_bad / tot_bad,
                "cum_good_pct": cum_good / tot_good,
            }
        )
    out = pd.DataFrame(rows)
    out["ks"] = (out["cum_bad_pct"] - out["cum_good_pct"]).abs()
    return out


def bootstrap_auc_ci(
    y: np.ndarray,
    s: np.ndarray,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for AUC: (auc, lower, upper).  Seeded.

    Resamples with replacement; draws with a single class are redrawn.
    """
    ya = _check_binary(y)
    sa = np.asarray(s, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    n = len(ya)
    aucs = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, n, n)
        yb = ya[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        aucs.append(roc_auc(yb, sa[idx]))
    lo, hi = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return roc_auc(ya, sa), float(lo), float(hi)


def brier_score(y: np.ndarray, pd_hat: np.ndarray) -> float:
    """Mean squared error of predicted PDs against outcomes."""
    ya = np.asarray(y, dtype=float).ravel()
    pa = np.asarray(pd_hat, dtype=float).ravel()
    if len(ya) != len(pa) or len(ya) == 0:
        raise ValueError("y and pd_hat must be same nonzero length")
    return float(np.mean((pa - ya) ** 2))


def hosmer_lemeshow(
    y: np.ndarray, pd_hat: np.ndarray, n_groups: int = 10
) -> tuple[float, float, pd.DataFrame]:
    """Hosmer-Lemeshow goodness-of-fit test on predicted-PD deciles.

    Returns (chi2 statistic, p-value with df = groups - 2, group table with
    observed vs expected defaults).
    """
    ya = np.asarray(y, dtype=float).ravel()
    pa = np.asarray(pd_hat, dtype=float).ravel()
    if len(ya) != len(pa) or len(ya) == 0:
        raise ValueError("y and pd_hat must be same nonzero length")
    groups = pd.qcut(pd.Series(pa).rank(method="first"), n_groups, labels=False)
    rows = []
    chi2 = 0.0
    for g in range(n_groups):
        m = (groups == g).to_numpy()
        n_g = int(m.sum())
        obs1 = float(ya[m].sum())
        exp1 = float(pa[m].sum())
        obs0, exp0 = n_g - obs1, n_g - exp1
        if exp1 > 0 and exp0 > 0:
            chi2 += (obs1 - exp1) ** 2 / exp1 + (obs0 - exp0) ** 2 / exp0
        rows.append(
            {
                "group": g + 1,
                "n": n_g,
                "mean_pd": float(pa[m].mean()),
                "observed_defaults": obs1,
                "expected_defaults": exp1,
                "observed_rate": obs1 / n_g,
            }
        )
    df_ = max(n_groups - 2, 1)
    p_value = float(stats.chi2.sf(chi2, df_))
    return float(chi2), p_value, pd.DataFrame(rows)


def calibration_table(
    y: np.ndarray, pd_hat: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability table: mean predicted PD vs observed default rate per bin."""
    _, _, table = hosmer_lemeshow(y, pd_hat, n_groups=n_bins)
    table = table.rename(columns={"mean_pd": "predicted_pd"})
    return table[["group", "n", "predicted_pd", "observed_rate"]]


def psi_from_proportions(
    p_expected: np.ndarray, p_actual: np.ndarray, eps: float = 1e-6
) -> float:
    """PSI from per-bin proportions: sum (p_act - p_exp) * ln(p_act / p_exp).

    Zero proportions are floored at ``eps`` (standard practice).  Proportions
    must each sum to ~1.
    """
    pe = np.asarray(p_expected, dtype=float)
    pa = np.asarray(p_actual, dtype=float)
    if pe.shape != pa.shape:
        raise ValueError("proportion vectors must have the same shape")
    if not (np.isclose(pe.sum(), 1.0, atol=1e-6) and np.isclose(pa.sum(), 1.0, atol=1e-6)):
        raise ValueError("proportions must each sum to 1")
    pe = np.maximum(pe, eps)
    pa = np.maximum(pa, eps)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    breaks: np.ndarray | None = None,
) -> float:
    """PSI between two continuous samples.

    Bins are deciles of the ``expected`` (baseline) sample unless ``breaks``
    is given.  NaNs form their own bin in both samples.
    """
    ea = np.asarray(expected, dtype=float).ravel()
    aa = np.asarray(actual, dtype=float).ravel()
    if len(ea) == 0 or len(aa) == 0:
        raise ValueError("empty sample")
    e_na, a_na = np.isnan(ea), np.isnan(aa)
    ev, av = ea[~e_na], aa[~a_na]
    if breaks is None:
        breaks = np.unique(np.quantile(ev, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e_idx = np.searchsorted(breaks, ev, side="left")
    a_idx = np.searchsorted(breaks, av, side="left")
    k = len(breaks) + 1
    e_counts = np.bincount(e_idx, minlength=k).astype(float)
    a_counts = np.bincount(a_idx, minlength=k).astype(float)
    if e_na.any() or a_na.any():
        e_counts = np.append(e_counts, float(e_na.sum()))
        a_counts = np.append(a_counts, float(a_na.sum()))
    return psi_from_proportions(e_counts / len(ea), a_counts / len(aa))


def psi_status(value: float) -> str:
    """Classify PSI: 'stable' (<0.10), 'monitor' (0.10-0.25), 'shifted' (>0.25)."""
    if value < 0.10:
        return "stable"
    if value <= 0.25:
        return "monitor"
    return "shifted"


def psi_report(
    train: pd.DataFrame, oot: pd.DataFrame, cols: list[str], n_bins: int = 10
) -> pd.DataFrame:
    """Per-feature PSI (train vs OOT) with status labels."""
    rows = []
    for c in cols:
        v = psi(train[c].to_numpy(dtype=float), oot[c].to_numpy(dtype=float), n_bins)
        rows.append({"feature": c, "psi": v, "status": psi_status(v)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def decile_table(y: np.ndarray, pd_hat: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Rank-ordering table: observed default rate by predicted-PD decile
    (decile 1 = safest).  A healthy model has monotone increasing rates."""
    ya = _check_binary(y)
    pa = np.asarray(pd_hat, dtype=float).ravel()
    groups = pd.qcut(pd.Series(pa).rank(method="first"), n_bins, labels=False)
    rows = []
    for g in range(n_bins):
        m = (groups == g).to_numpy()
        rows.append(
            {
                "decile": g + 1,
                "n": int(m.sum()),
                "mean_pd": float(pa[m].mean()),
                "default_rate": float(ya[m].mean()),
            }
        )
    return pd.DataFrame(rows)


def is_monotone(values: np.ndarray, increasing: bool = True, tol: float = 0.0) -> bool:
    """True if the sequence is monotone (within ``tol`` per step)."""
    d = np.diff(np.asarray(values, dtype=float))
    return bool(np.all(d >= -tol)) if increasing else bool(np.all(d <= tol))
