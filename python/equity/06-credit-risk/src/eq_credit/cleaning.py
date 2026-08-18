"""Typed cleaning pipeline for the scorecard: leakage guard, winsorization,
duplicates, missing-value strategies and the train/OOT temporal split.

Missing-value philosophy
------------------------
The scorecard-native treatment of missing values is to keep them: WOE binning
assigns *missing its own bin*, so informative missingness (e.g. thin-file
borrowers with no payment history) becomes signal rather than bias.  Median
fill is provided ONLY for benchmark models (e.g. a raw-feature logit) that
cannot handle NaN.

Leakage guard
-------------
``FORBIDDEN_POST_OUTCOME_FIELDS`` is the documented list of fields that are
functions of the default outcome (write-offs, recoveries, arrears realised
after origination).  :func:`check_leakage` refuses any feature list touching
them — this is a hard control, mirroring SR 11-7-style model risk controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "FORBIDDEN_POST_OUTCOME_FIELDS",
    "LeakageError",
    "check_leakage",
    "find_duplicates",
    "drop_duplicate_loans",
    "WinsorBounds",
    "fit_winsor_bounds",
    "apply_winsor",
    "MedianImputer",
    "train_oot_split",
]

#: Fields knowable only after the default outcome.  Using any of them as a
#: model feature is target leakage.
FORBIDDEN_POST_OUTCOME_FIELDS: tuple[str, ...] = (
    "writeoff_flag",
    "recovery_amount",
    "days_past_due_max",
    "default_date",
    "collections_fees",
    "charge_off_amount",
)


class LeakageError(ValueError):
    """Raised when a candidate feature list contains post-outcome fields."""


def check_leakage(
    feature_cols: list[str] | tuple[str, ...],
    forbidden: tuple[str, ...] = FORBIDDEN_POST_OUTCOME_FIELDS,
) -> None:
    """Raise :class:`LeakageError` if any feature is a forbidden post-outcome field.

    Parameters
    ----------
    feature_cols : list of str
        Candidate model features.
    forbidden : tuple of str
        Deny-list; defaults to :data:`FORBIDDEN_POST_OUTCOME_FIELDS`.

    Raises
    ------
    LeakageError
        Listing the offending columns.
    """
    bad = sorted(set(feature_cols) & set(forbidden))
    if bad:
        raise LeakageError(
            f"Post-outcome (leaky) fields in feature list: {bad}. "
            "These are knowable only after the default outcome and must not "
            "be used as model inputs."
        )


def find_duplicates(df: pd.DataFrame, key: str = "loan_id") -> pd.DataFrame:
    """Return all rows participating in a duplicated ``key`` (all occurrences)."""
    if key not in df.columns:
        raise ValueError(f"key column {key!r} not in DataFrame")
    return df[df.duplicated(subset=[key], keep=False)].sort_values(key)


def drop_duplicate_loans(df: pd.DataFrame, key: str = "loan_id") -> pd.DataFrame:
    """Drop duplicate rows by ``key``, keeping the first occurrence."""
    if key not in df.columns:
        raise ValueError(f"key column {key!r} not in DataFrame")
    return df.drop_duplicates(subset=[key], keep="first").reset_index(drop=True)


@dataclass(frozen=True)
class WinsorBounds:
    """Per-column winsorization bounds fitted on the training sample.

    Attributes
    ----------
    lower, upper : dict[str, float]
        Column -> clip bound (quantiles of the training distribution).
    lower_q, upper_q : float
        Quantile levels used at fit time.
    """

    lower: dict[str, float] = field(default_factory=dict)
    upper: dict[str, float] = field(default_factory=dict)
    lower_q: float = 0.01
    upper_q: float = 0.99


def fit_winsor_bounds(
    df: pd.DataFrame,
    cols: list[str],
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> WinsorBounds:
    """Fit winsorization bounds (train-sample quantiles), ignoring NaN.

    Raises
    ------
    ValueError
        If quantiles are not ordered ``0 <= lower_q < upper_q <= 1``.
    """
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("require 0 <= lower_q < upper_q <= 1")
    lower = {c: float(df[c].quantile(lower_q)) for c in cols}
    upper = {c: float(df[c].quantile(upper_q)) for c in cols}
    return WinsorBounds(lower=lower, upper=upper, lower_q=lower_q, upper_q=upper_q)


def apply_winsor(df: pd.DataFrame, bounds: WinsorBounds) -> pd.DataFrame:
    """Clip columns to fitted bounds (NaN passes through untouched)."""
    out = df.copy()
    for c, lo in bounds.lower.items():
        out[c] = out[c].clip(lower=lo, upper=bounds.upper[c])
    return out


@dataclass
class MedianImputer:
    """Median fill for benchmark models only (NOT the scorecard path).

    The scorecard keeps NaN so WOE binning can assign missing its own bin.
    """

    medians: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> "MedianImputer":
        """Store training medians (NaN ignored) for ``cols``."""
        self.medians = {c: float(df[c].median()) for c in cols}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with stored training medians."""
        if not self.medians:
            raise ValueError("MedianImputer.transform called before fit")
        out = df.copy()
        for c, m in self.medians.items():
            out[c] = out[c].fillna(m)
        return out


def train_oot_split(
    df: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    date_col: str = "origination_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Temporal train / out-of-time split at ``cutoff``.

    Rows with ``date_col`` strictly before ``cutoff`` go to train; the rest to
    OOT.  Guarantees ``max(train date) < cutoff <= min(oot date)`` so no
    observation can leak across the boundary.

    Raises
    ------
    ValueError
        If either side of the split would be empty.
    """
    if date_col not in df.columns:
        raise ValueError(f"date column {date_col!r} not in DataFrame")
    cutoff_ts = pd.to_datetime(cutoff)
    dates = pd.to_datetime(df[date_col])
    train = df[dates < cutoff_ts].reset_index(drop=True)
    oot = df[dates >= cutoff_ts].reset_index(drop=True)
    if len(train) == 0 or len(oot) == 0:
        raise ValueError(
            f"train/OOT split at {cutoff_ts.date()} leaves an empty side "
            f"(train={len(train)}, oot={len(oot)})"
        )
    assert pd.to_datetime(train[date_col]).max() < cutoff_ts
    assert pd.to_datetime(oot[date_col]).min() >= cutoff_ts
    return train, oot
