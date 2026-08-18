"""Panel-aware cleaning, leakage guards, and time/country splits for sovereign panels.

Why a random row split is wrong here
------------------------------------
The panel is country-year.  Macro features are strongly serially correlated
within a country (AR(1)-type dynamics, structural country effects), and the
crisis outcome clusters in country spells and regional contagion years.  A
random row split therefore places 2016 GreeceX in the training set and 2015
GreeceX in the test set: the model has effectively *seen* the test row (same
country effect, nearly identical features) and validation metrics are
optimistically biased.  Worse, contagion years straddle the split, so crisis
information leaks across it.  The correct designs are:

* **out-of-time split** (`time_split`): train on years <= T, test strictly on
  years > T — this is how the model will actually be used (fit on history,
  predict the future); and
* **country-holdout split** (`country_holdout_split`): entire countries held
  out, testing cross-sectional generalisation to unseen sovereigns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data.synthetic import LEAKY_FIELDS

__all__ = [
    "LEAKY_FIELDS",
    "clean_panel",
    "drop_leaky_fields",
    "assert_no_leaky_fields",
    "time_split",
    "country_holdout_split",
]

_REQUIRED = ("country", "year", "default")

#: Physically-impossible value bounds; violations are set to NaN (not clipped,
#: so the WOE missing bin can pick them up honestly).
_VALID_RANGES: dict[str, tuple[float, float]] = {
    "reserves_import_cover": (0.0, 60.0),
    "ext_debt_gdp": (0.0, 1000.0),
    "st_debt_reserves": (0.0, 100.0),
    "ca_gdp": (-100.0, 100.0),
    "fiscal_gdp": (-100.0, 100.0),
    "inflation": (-30.0, 100_000.0),
    "political_stability": (-2.5, 2.5),
}


def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a sovereign country-year panel (panel-aware, non-destructive to NaN).

    Steps: validate required columns; drop exact duplicate country-year rows
    (keeping the first); replace +/-inf with NaN; null out physically
    impossible values per ``_VALID_RANGES``; coerce ``year`` to int and the
    outcome to {0,1}; sort by (country, year).

    Missing values are deliberately *kept* — the WOE layer assigns them an
    explicit missing bin, which is itself informative (data gaps correlate
    with weak institutions).

    Raises
    ------
    ValueError
        If required columns are absent or the outcome is not binary.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    out = df.copy()
    out = out.drop_duplicates(subset=["country", "year"], keep="first")
    out = out.replace([np.inf, -np.inf], np.nan)
    for col, (lo, hi) in _VALID_RANGES.items():
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            out[col] = vals.where((vals >= lo) & (vals <= hi))
    out["year"] = out["year"].astype(int)
    outcome = pd.to_numeric(out["default"], errors="raise")
    if not set(np.unique(outcome.dropna())) <= {0, 1}:
        raise ValueError("outcome column 'default' must be binary {0,1}")
    out["default"] = outcome.astype(int)
    return out.sort_values(["country", "year"]).reset_index(drop=True)


def drop_leaky_fields(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Drop known post-outcome ('post-crisis') fields from the modelling frame.

    ``imf_program_next_year`` and ``devaluation_next_year_pct`` are recorded
    *after* the crisis outcome and are near-deterministic functions of it;
    including them yields a scorecard with perfect in-sample discrimination
    and zero forecasting content.
    """
    cols = [c for c in (*LEAKY_FIELDS, *extra) if c in df.columns]
    return df.drop(columns=cols)


def assert_no_leaky_fields(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> None:
    """Raise ``ValueError`` if any known leaky field is still present.

    Call this immediately before model fitting as a hard guard.
    """
    present = [c for c in (*LEAKY_FIELDS, *extra) if c in df.columns]
    if present:
        raise ValueError(
            f"leaky post-outcome fields present in modelling frame: {present}; "
            "call drop_leaky_fields() before fitting"
        )


def time_split(
    df: pd.DataFrame,
    train_end_year: int,
    test_start_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split: train on years <= ``train_end_year``, test on later years.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel with a ``year`` column.
    train_end_year : int
        Last year included in training.
    test_start_year : int, optional
        First test year; defaults to ``train_end_year + 1``.  Must be
        strictly greater than ``train_end_year`` (an overlap would leak).

    Returns
    -------
    (train, test) : tuple of DataFrame
        Guaranteed: ``train.year.max() < test.year.min()`` (when both are
        non-empty) — no same-country future row can ever land in train.
    """
    if test_start_year is None:
        test_start_year = train_end_year + 1
    if test_start_year <= train_end_year:
        raise ValueError(
            f"test_start_year ({test_start_year}) must be > train_end_year "
            f"({train_end_year}); overlapping windows leak future information"
        )
    train = df[df["year"] <= train_end_year].copy()
    test = df[df["year"] >= test_start_year].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def country_holdout_split(
    df: pd.DataFrame,
    holdout_frac: float = 0.2,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country-holdout split: entire countries assigned to the test set.

    Complements ``time_split``: tests generalisation to sovereigns never seen
    in training (no shared country effect).

    Parameters
    ----------
    holdout_frac : float
        Fraction of countries (not rows) held out, in (0, 1).
    seed : int
        Seed for the country sampling.
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError("holdout_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    countries = np.sort(df["country"].unique())
    n_hold = max(1, int(round(holdout_frac * len(countries))))
    held = set(rng.choice(countries, size=n_hold, replace=False))
    test_mask = df["country"].isin(held)
    return (
        df[~test_mask].reset_index(drop=True),
        df[test_mask].reset_index(drop=True),
    )
