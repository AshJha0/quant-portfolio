"""Weight-of-Evidence / Information-Value binning with missing bin and monotone merge.

Conventions
-----------
For bin *i* with ``n_bad`` events (defaults) and ``n_good`` non-events:

.. math::

    \\mathrm{WOE}_i = \\ln\\frac{n^{bad}_i / N^{bad}}{n^{good}_i / N^{good}},
    \\qquad
    \\mathrm{IV} = \\sum_i \\left(\\frac{n^{bad}_i}{N^{bad}} -
    \\frac{n^{good}_i}{N^{good}}\\right)\\mathrm{WOE}_i .

Positive WOE = riskier than average (bad-over-good convention; sign is a pure
convention — flip both the ratio and the logistic coefficients to get the
scorecard-vendor convention).  A ``smoothing`` count (default 0.5, Laplace
style) is added to each bin's bad and good counts so empty cells stay finite;
set ``smoothing=0`` for exact textbook arithmetic.

Missing values get their **own explicit bin**: in sovereign data, a missing
governance score is itself a signal (weak statistical capacity correlates
with weak institutions), and silently imputing it would both bias the model
and hide a data-quality problem.

Leakage red-flag: a single feature with IV above ~1.0 almost never happens
with honest ex-ante macro data; it is the classic signature of a
post-outcome field.  ``flag_leaky_iv`` encodes that rule.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

__all__ = [
    "WOEBin",
    "WOETable",
    "woe_table",
    "monotone_merge",
    "woe_transform",
    "iv_report",
    "flag_leaky_iv",
]


@dataclass(frozen=True)
class WOEBin:
    """A single WOE bin. ``left``/``right`` are NaN for the missing bin."""

    label: str
    left: float
    right: float
    is_missing: bool
    n: int
    n_bad: int
    n_good: int
    woe: float
    iv: float

    @property
    def bad_rate(self) -> float:
        return self.n_bad / self.n if self.n > 0 else np.nan


@dataclass(frozen=True)
class WOETable:
    """WOE binning of one feature against a binary outcome."""

    feature: str
    bins: tuple[WOEBin, ...]
    bad_total: int
    good_total: int
    smoothing: float

    @property
    def iv(self) -> float:
        """Total information value (sum of per-bin contributions)."""
        return float(sum(b.iv for b in self.bins))

    @property
    def numeric_bins(self) -> tuple[WOEBin, ...]:
        return tuple(b for b in self.bins if not b.is_missing)

    @property
    def missing_bin(self) -> WOEBin | None:
        for b in self.bins:
            if b.is_missing:
                return b
        return None

    def to_frame(self) -> pd.DataFrame:
        """Human-readable binning table (one row per bin)."""
        return pd.DataFrame(
            [
                {
                    "bin": b.label,
                    "n": b.n,
                    "n_bad": b.n_bad,
                    "bad_rate": b.bad_rate,
                    "woe": b.woe,
                    "iv": b.iv,
                }
                for b in self.bins
            ]
        )


def _make_bin(
    label: str,
    left: float,
    right: float,
    is_missing: bool,
    n_bad: int,
    n_good: int,
    bad_total: int,
    good_total: int,
    smoothing: float,
) -> WOEBin:
    if bad_total <= 0 or good_total <= 0:
        raise ValueError("need at least one bad and one good observation overall")
    bs = (n_bad + smoothing) / (bad_total + 2 * smoothing)
    gs = (n_good + smoothing) / (good_total + 2 * smoothing)
    if bs <= 0 or gs <= 0:
        raise ValueError(
            f"bin {label!r} has an empty cell with smoothing=0; "
            "use smoothing>0 or coarser bins"
        )
    woe = float(np.log(bs / gs))
    iv = float((bs - gs) * woe)
    return WOEBin(label, left, right, is_missing, n_bad + n_good, n_bad, n_good, woe, iv)


def woe_table(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    n_bins: int = 5,
    feature: str = "x",
    smoothing: float = 0.5,
    edges: np.ndarray | None = None,
) -> WOETable:
    """Bin a numeric feature by (approximate) quantiles and compute WOE/IV.

    Parameters
    ----------
    x : array-like
        Numeric feature; NaN values go to an explicit missing bin.
    y : array-like
        Binary outcome (1 = bad/default).
    n_bins : int
        Target number of quantile bins (duplicated edges are collapsed, so
        the realised count can be lower for coarse features).
    feature : str
        Name recorded on the table.
    smoothing : float
        Laplace count added to each bin's bad and good tallies (0 = exact).
    edges : ndarray, optional
        Explicit *interior* + outer bin edges (length k+1, ascending).  When
        given, ``n_bins`` is ignored.  Bins are ``(e[i], e[i+1]]`` with the
        first bin closed on the left.

    Returns
    -------
    WOETable
    """
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("y must be binary {0,1}")
    bad_total = int(y.sum())
    good_total = int(len(y) - bad_total)

    miss = np.isnan(x)
    xv, yv = x[~miss], y[~miss]
    if xv.size == 0:
        raise ValueError("feature is entirely missing")

    if edges is None:
        uniq = np.unique(xv)
        if uniq.size == 1:
            edges = np.array([uniq[0], uniq[0]])
        elif uniq.size <= n_bins:
            # low-cardinality feature (e.g. a 0/1 regime dummy): one bin per
            # distinct value, split at midpoints — quantile edges would
            # collapse a dominant value into a single degenerate bin
            edges = np.r_[uniq[0], (uniq[:-1] + uniq[1:]) / 2.0, uniq[-1]]
        else:
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.unique(np.quantile(xv, qs))
            if edges.size < 3:  # extreme skew collapsed the quantiles
                edges = np.unique(np.r_[uniq[0], np.median(xv), uniq[-1]])
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        edges = np.array([edges[0], edges[0]])

    k = edges.size - 1
    # assign: bin i covers (edges[i], edges[i+1]], first bin closed left
    idx = np.searchsorted(edges[1:-1], xv, side="left")
    bins: list[WOEBin] = []
    for i in range(k):
        sel = idx == i
        nb = int(yv[sel].sum())
        ng = int(sel.sum() - nb)
        lo, hi = edges[i], edges[i + 1]
        label = f"({lo:.4g}, {hi:.4g}]" if i > 0 else f"[{lo:.4g}, {hi:.4g}]"
        bins.append(
            _make_bin(label, lo, hi, False, nb, ng, bad_total, good_total, smoothing)
        )
    if miss.any():
        nb = int(y[miss].sum())
        ng = int(miss.sum() - nb)
        bins.append(
            _make_bin("missing", np.nan, np.nan, True, nb, ng, bad_total, good_total, smoothing)
        )
    return WOETable(feature, tuple(bins), bad_total, good_total, smoothing)


def monotone_merge(table: WOETable) -> WOETable:
    """Merge adjacent numeric bins until WOE is monotone in the bin order.

    The target direction is the sign of the correlation between bin index
    and WOE on the input table (i.e. the dominant trend).  Adjacent bins
    violating that direction are pooled and WOE/IV recomputed, repeatedly,
    until monotone.  The missing bin is never merged — missingness has no
    natural order.

    Monotone WOE is required for a scorecard: risk managers must be able to
    say "less reserve cover => more points of risk" without reversals caused
    by binning noise.
    """
    nums = list(table.numeric_bins)
    if len(nums) <= 2:
        return table
    woes = np.array([b.woe for b in nums])
    direction = 1.0 if np.corrcoef(np.arange(len(nums)), woes)[0, 1] >= 0 else -1.0

    def rebuild(nb: int, ng: int, left: float, right: float, first: bool) -> WOEBin:
        label = f"({left:.4g}, {right:.4g}]" if not first else f"[{left:.4g}, {right:.4g}]"
        return _make_bin(label, left, right, False, nb, ng,
                         table.bad_total, table.good_total, table.smoothing)

    while len(nums) > 1:
        woes = np.array([b.woe for b in nums])
        diffs = direction * np.diff(woes)
        bad_idx = np.where(diffs < 0)[0]
        if bad_idx.size == 0:
            break
        i = int(bad_idx[0])
        a, b = nums[i], nums[i + 1]
        merged = rebuild(a.n_bad + b.n_bad, a.n_good + b.n_good, a.left, b.right, i == 0)
        nums = nums[:i] + [merged] + nums[i + 2:]

    out_bins = tuple(nums) + ((table.missing_bin,) if table.missing_bin else ())
    return replace(table, bins=out_bins)


def woe_transform(x: np.ndarray | pd.Series, table: WOETable) -> np.ndarray:
    """Map raw feature values to their bin WOE.

    NaN maps to the missing-bin WOE (0.0 if the table has no missing bin —
    neutral for unseen missingness).  Out-of-range values are clamped to the
    first/last numeric bin.
    """
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    nums = table.numeric_bins
    if not nums:
        raise ValueError("table has no numeric bins")
    edges = np.array([b.right for b in nums[:-1]])
    woes = np.array([b.woe for b in nums])
    out = np.empty_like(x)
    miss = np.isnan(x)
    idx = np.searchsorted(edges, x[~miss], side="left")
    out[~miss] = woes[idx]
    mb = table.missing_bin
    out[miss] = mb.woe if mb is not None else 0.0
    return out


def iv_report(
    df: pd.DataFrame,
    features: list[str],
    outcome: str = "default",
    n_bins: int = 5,
    smoothing: float = 0.5,
) -> pd.DataFrame:
    """IV of each feature (after monotone merge), sorted descending.

    Returns a DataFrame with columns ``feature, iv, n_bins, leaky_flag``.
    """
    rows = []
    for f in features:
        t = monotone_merge(woe_table(df[f], df[outcome], n_bins=n_bins,
                                     feature=f, smoothing=smoothing))
        rows.append({"feature": f, "iv": t.iv, "n_bins": len(t.bins)})
    rep = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    rep["leaky_flag"] = rep["iv"] > 1.0
    return rep


def flag_leaky_iv(iv_by_feature: dict[str, float], threshold: float = 1.0) -> list[str]:
    """Features whose IV exceeds ``threshold`` — near-certain target leakage.

    Empirical rule of thumb: IV of 0.3-0.5 is already "suspiciously strong"
    for macro data; above ~1.0 the 'feature' is effectively an outcome
    recording (e.g. an IMF-programme flag stamped after the crisis).
    """
    return sorted([f for f, iv in iv_by_feature.items() if iv > threshold])
