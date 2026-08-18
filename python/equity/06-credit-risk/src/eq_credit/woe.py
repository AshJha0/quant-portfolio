"""Supervised WOE binning and Information Value.

Conventions (industry standard):

* ``WOE_i = ln( dist_good_i / dist_bad_i )`` where ``dist_good_i`` is the
  share of all GOODS (non-defaults) falling in bin i and ``dist_bad_i`` the
  share of all BADS (defaults).  Positive WOE = safer-than-average bin.
* ``IV = sum_i (dist_good_i - dist_bad_i) * WOE_i``  (always >= 0).
* IV strength thresholds (Siddiqi):  < 0.02 useless, 0.02-0.1 weak,
  0.1-0.3 medium, 0.3-0.5 strong, > 0.5 **suspicious** — flagged as a
  possible leakage via :class:`SuspiciousIVWarning`.

Numeric features are pre-binned on quantiles, then adjacent bins are merged
(smallest chi-square first — i.e. the statistically most similar pair) until
the bad rate is monotone in the bin order; missing values get their own bin.
Zero-count cells are smoothed (adds 0.5 to both good and bad counts in that
bin only) so WOE is never +/-inf; bins with non-zero counts are exact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "SuspiciousIVWarning",
    "IV_THRESHOLDS",
    "iv_strength",
    "woe_iv_from_counts",
    "FeatureBinning",
    "fit_numeric_binning",
    "fit_categorical_binning",
    "WOETransformer",
]

#: (upper_bound, label) IV interpretation thresholds.
IV_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.02, "useless"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (np.inf, "suspicious"),
)


class SuspiciousIVWarning(UserWarning):
    """IV > 0.5: too good to be true — investigate for target leakage."""


def iv_strength(iv: float) -> str:
    """Map an IV value to its standard interpretation label."""
    if iv < 0:
        raise ValueError("IV cannot be negative")
    for ub, label in IV_THRESHOLDS:
        if iv < ub:
            return label
    return "suspicious"  # pragma: no cover


def woe_iv_from_counts(
    n_good: np.ndarray, n_bad: np.ndarray, smoothing: float = 0.5
) -> tuple[np.ndarray, np.ndarray, float]:
    """WOE and IV from per-bin good/bad counts.

    ``smoothing`` is added to BOTH counts of a bin only when that bin has a
    zero good or zero bad count, so hand-computed values on tables with no
    zero cells are exact.

    Parameters
    ----------
    n_good, n_bad : np.ndarray
        Per-bin counts of goods (y=0) and bads (y=1).
    smoothing : float
        Zero-cell correction (default 0.5).

    Returns
    -------
    woe : np.ndarray
        Per-bin WOE = ln(dist_good / dist_bad).
    iv_contrib : np.ndarray
        Per-bin IV contribution (dist_good - dist_bad) * WOE.
    iv : float
        Total Information Value.
    """
    n_good = np.asarray(n_good, dtype=float)
    n_bad = np.asarray(n_bad, dtype=float)
    if n_good.shape != n_bad.shape:
        raise ValueError("n_good and n_bad must have the same shape")
    tot_good, tot_bad = n_good.sum(), n_bad.sum()
    if tot_good <= 0 or tot_bad <= 0:
        raise ValueError(
            "WOE requires both goods and bads in the sample "
            f"(goods={tot_good:.0f}, bads={tot_bad:.0f})"
        )
    g = n_good.copy()
    b = n_bad.copy()
    zero = (g == 0) | (b == 0)
    g[zero] += smoothing
    b[zero] += smoothing
    dist_good = g / tot_good
    dist_bad = b / tot_bad
    woe = np.log(dist_good / dist_bad)
    iv_contrib = (dist_good - dist_bad) * woe
    return woe, iv_contrib, float(iv_contrib.sum())


@dataclass
class FeatureBinning:
    """Fitted WOE binning for one feature.

    Attributes
    ----------
    feature : str
        Feature name.
    kind : str
        ``"numeric"`` or ``"categorical"``.
    edges : np.ndarray
        For numeric features: interior cut points; bin i is
        ``(edges[i-1], edges[i]]`` with open outer bins.
    categories : list[list[str]]
        For categorical features: category groups per bin.
    woes : np.ndarray
        WOE per (non-missing) bin.
    missing_woe : float
        WOE of the missing bin (0.0 if no missing observed at fit).
    has_missing_bin : bool
        Whether missing values were present at fit time.
    iv : float
        Total IV including the missing bin.
    table : pd.DataFrame
        Binning report: bin label, counts, bad rate, WOE, IV contribution.
    """

    feature: str
    kind: str
    woes: np.ndarray
    iv: float
    table: pd.DataFrame
    edges: np.ndarray = field(default_factory=lambda: np.array([]))
    categories: list[list[str]] = field(default_factory=list)
    missing_woe: float = 0.0
    has_missing_bin: bool = False

    def transform(self, x: pd.Series | np.ndarray) -> np.ndarray:
        """Map raw values to WOE; NaN maps to the missing-bin WOE."""
        arr = np.asarray(pd.Series(x), dtype=object)
        out = np.empty(len(arr), dtype=float)
        if self.kind == "numeric":
            vals = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)
            isna = np.isnan(vals)
            idx = np.searchsorted(self.edges, vals[~isna], side="left")
            out[~isna] = self.woes[idx]
            out[isna] = self.missing_woe
        else:
            cat_to_bin: dict[str, int] = {
                c: i for i, group in enumerate(self.categories) for c in group
            }
            for j, v in enumerate(arr):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    out[j] = self.missing_woe
                else:
                    out[j] = self.woes[cat_to_bin.get(str(v), 0)] if str(v) in cat_to_bin else self.missing_woe
        return out


def _chi2_adjacent(g: np.ndarray, b: np.ndarray, i: int) -> float:
    """2x2 chi-square statistic between adjacent bins i and i+1."""
    obs = np.array([[g[i], b[i]], [g[i + 1], b[i + 1]]], dtype=float)
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    tot = obs.sum()
    if tot == 0 or (row == 0).any() or (col == 0).any():
        return 0.0
    exp = row @ col / tot
    return float(((obs - exp) ** 2 / exp).sum())


def _merge_pair(edges: list[float], g: list[float], b: list[float], i: int) -> None:
    g[i] += g[i + 1]
    b[i] += b[i + 1]
    del g[i + 1], b[i + 1], edges[i]


def _is_monotone(rate: np.ndarray) -> bool:
    d = np.diff(rate)
    return bool(np.all(d >= 0) or np.all(d <= 0))


def fit_numeric_binning(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    feature: str = "x",
    n_prebins: int = 20,
    min_bin_frac: float = 0.05,
    monotone: bool = True,
    smoothing: float = 0.5,
) -> FeatureBinning:
    """Fit monotone WOE binning for a numeric feature.

    Algorithm: quantile pre-bins on non-missing values -> merge bins smaller
    than ``min_bin_frac`` of non-missing rows -> while the bad rate is not
    monotone across bins, merge the adjacent pair with the smallest 2x2
    chi-square (least evidence they differ).  Missing values form their own
    bin (excluded from the monotonicity constraint).

    Raises
    ------
    ValueError
        If the target has no defaults or no goods.
    """
    xs = pd.Series(x).astype(float).reset_index(drop=True)
    ys = pd.Series(y).astype(int).reset_index(drop=True)
    if len(xs) != len(ys):
        raise ValueError("x and y must have the same length")
    if ys.sum() == 0:
        raise ValueError(
            f"feature {feature!r}: zero defaults in sample — WOE/IV undefined; "
            "collect more data or use a low-default-portfolio approach"
        )
    if ys.sum() == len(ys):
        raise ValueError(f"feature {feature!r}: sample contains no goods (all default)")

    isna = xs.isna().to_numpy()
    xv = xs[~isna].to_numpy()
    yv = ys[~isna].to_numpy()
    if len(xv) == 0:
        raise ValueError(f"feature {feature!r} is entirely missing")

    qs = np.quantile(xv, np.linspace(0, 1, n_prebins + 1)[1:-1])
    edges = list(np.unique(qs))
    idx = np.searchsorted(np.array(edges), xv, side="left")
    n_bins = len(edges) + 1
    g = [float(np.sum((idx == i) & (yv == 0))) for i in range(n_bins)]
    b = [float(np.sum((idx == i) & (yv == 1))) for i in range(n_bins)]

    # Enforce minimum bin size.
    min_n = min_bin_frac * len(xv)
    while len(g) > 1:
        tot = np.array(g) + np.array(b)
        small = int(np.argmin(tot))
        if tot[small] >= min_n:
            break
        if small == 0:
            j = 0
        elif small == len(g) - 1:
            j = small - 1
        else:
            j = small - 1 if _chi2_adjacent(np.array(g), np.array(b), small - 1) <= \
                _chi2_adjacent(np.array(g), np.array(b), small) else small
        _merge_pair(edges, g, b, j)

    # Enforce monotone bad rate by merging the most-similar adjacent pair.
    if monotone:
        while len(g) > 2:
            ga, ba = np.array(g), np.array(b)
            rate = ba / np.maximum(ga + ba, 1.0)
            if _is_monotone(rate):
                break
            viol = [i for i in range(len(g) - 1)]
            chis = [_chi2_adjacent(ga, ba, i) for i in viol]
            _merge_pair(edges, g, b, int(np.argmin(chis)))

    ga, ba = np.array(g), np.array(b)
    n_missing = int(isna.sum())
    if n_missing > 0:
        gm = float(np.sum(ys[isna] == 0))
        bm = float(np.sum(ys[isna] == 1))
        all_g = np.append(ga, gm)
        all_b = np.append(ba, bm)
    else:
        all_g, all_b = ga, ba

    woe_all, ivc_all, iv = woe_iv_from_counts(all_g, all_b, smoothing=smoothing)
    woes = woe_all[: len(ga)]
    missing_woe = float(woe_all[-1]) if n_missing > 0 else 0.0

    labels = []
    e = [-np.inf] + list(edges) + [np.inf]
    for i in range(len(ga)):
        labels.append(f"({e[i]:.4g}, {e[i+1]:.4g}]")
    if n_missing > 0:
        labels.append("MISSING")
    table = pd.DataFrame(
        {
            "bin": labels,
            "n": (all_g + all_b).astype(int),
            "n_bad": all_b.astype(int),
            "bad_rate": all_b / np.maximum(all_g + all_b, 1.0),
            "woe": woe_all,
            "iv_contrib": ivc_all,
        }
    )
    fb = FeatureBinning(
        feature=feature,
        kind="numeric",
        woes=woes,
        iv=iv,
        table=table,
        edges=np.array(edges, dtype=float),
        missing_woe=missing_woe,
        has_missing_bin=n_missing > 0,
    )
    _warn_if_suspicious(feature, iv)
    return fb


def fit_categorical_binning(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    feature: str = "x",
    min_bin_frac: float = 0.02,
    smoothing: float = 0.5,
) -> FeatureBinning:
    """Fit WOE binning for a categorical feature (one bin per category;
    categories rarer than ``min_bin_frac`` are pooled into ``OTHER``).
    """
    xs = pd.Series(x).astype(object).reset_index(drop=True)
    ys = pd.Series(y).astype(int).reset_index(drop=True)
    if ys.sum() == 0:
        raise ValueError(
            f"feature {feature!r}: zero defaults in sample — WOE/IV undefined"
        )
    if ys.sum() == len(ys):
        raise ValueError(f"feature {feature!r}: sample contains no goods (all default)")
    isna = xs.isna().to_numpy()
    xv = xs[~isna].astype(str)
    yv = ys[~isna].to_numpy()

    counts = xv.value_counts()
    rare = set(counts[counts < min_bin_frac * len(xv)].index)
    grouped = xv.where(~xv.isin(rare), "OTHER")
    cats = list(pd.unique(grouped))
    groups: list[list[str]] = []
    g_list, b_list = [], []
    for c in cats:
        mask = (grouped == c).to_numpy()
        members = sorted(rare) if c == "OTHER" else [c]
        groups.append([str(m) for m in members] if c == "OTHER" else [c])
        g_list.append(float(np.sum(yv[mask] == 0)))
        b_list.append(float(np.sum(yv[mask] == 1)))

    ga, ba = np.array(g_list), np.array(b_list)
    n_missing = int(isna.sum())
    if n_missing > 0:
        all_g = np.append(ga, float(np.sum(ys[isna] == 0)))
        all_b = np.append(ba, float(np.sum(ys[isna] == 1)))
    else:
        all_g, all_b = ga, ba
    woe_all, ivc_all, iv = woe_iv_from_counts(all_g, all_b, smoothing=smoothing)
    labels = [",".join(gr) if gr else "OTHER" for gr in groups]
    if n_missing > 0:
        labels.append("MISSING")
    table = pd.DataFrame(
        {
            "bin": labels,
            "n": (all_g + all_b).astype(int),
            "n_bad": all_b.astype(int),
            "bad_rate": all_b / np.maximum(all_g + all_b, 1.0),
            "woe": woe_all,
            "iv_contrib": ivc_all,
        }
    )
    fb = FeatureBinning(
        feature=feature,
        kind="categorical",
        woes=woe_all[: len(ga)],
        iv=iv,
        table=table,
        categories=groups,
        missing_woe=float(woe_all[-1]) if n_missing > 0 else 0.0,
        has_missing_bin=n_missing > 0,
    )
    _warn_if_suspicious(feature, iv)
    return fb


def _warn_if_suspicious(feature: str, iv: float) -> None:
    if iv > 0.5:
        warnings.warn(
            f"feature {feature!r} has IV = {iv:.3f} > 0.5 — suspiciously high; "
            "investigate for target leakage (post-outcome field?)",
            SuspiciousIVWarning,
            stacklevel=3,
        )


class WOETransformer:
    """Fit WOE binnings for a set of features and transform to WOE space.

    Parameters
    ----------
    numeric_features, categorical_features : list of str
        Feature lists (checked against the leakage deny-list by the caller).
    n_prebins, min_bin_frac, monotone : see :func:`fit_numeric_binning`.
    non_monotone_features : list of str, optional
        Numeric features exempt from the monotonicity constraint (e.g. a
        U-shaped liquidity ratio, where forcing monotone WOE would destroy
        the real risk pattern).
    """

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str] | None = None,
        n_prebins: int = 20,
        min_bin_frac: float = 0.05,
        monotone: bool = True,
        non_monotone_features: list[str] | None = None,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features or [])
        self.n_prebins = n_prebins
        self.min_bin_frac = min_bin_frac
        self.monotone = monotone
        self.non_monotone_features = list(non_monotone_features or [])
        self.binnings_: dict[str, FeatureBinning] = {}

    def fit(self, df: pd.DataFrame, target: str = "default") -> "WOETransformer":
        """Fit one binning per feature on ``df`` against ``target``."""
        y = df[target]
        for f in self.numeric_features:
            self.binnings_[f] = fit_numeric_binning(
                df[f], y, feature=f, n_prebins=self.n_prebins,
                min_bin_frac=self.min_bin_frac,
                monotone=self.monotone and f not in self.non_monotone_features,
            )
        for f in self.categorical_features:
            self.binnings_[f] = fit_categorical_binning(df[f], y, feature=f)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of WOE-transformed features (columns ``woe_<f>``)."""
        if not self.binnings_:
            raise ValueError("WOETransformer.transform called before fit")
        out = {}
        for f, fb in self.binnings_.items():
            out[f"woe_{f}"] = fb.transform(df[f])
        return pd.DataFrame(out, index=df.index)

    def iv_table(self) -> pd.DataFrame:
        """IV summary per feature with strength labels, sorted descending."""
        rows = [
            {"feature": f, "iv": fb.iv, "strength": iv_strength(fb.iv)}
            for f, fb in self.binnings_.items()
        ]
        return (
            pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
        )
