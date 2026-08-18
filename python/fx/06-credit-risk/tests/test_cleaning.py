"""Panel cleaning, leakage guards, time-based and country-holdout splits."""

import numpy as np
import pandas as pd
import pytest

from fx_credit.cleaning import (
    LEAKY_FIELDS,
    assert_no_leaky_fields,
    clean_panel,
    country_holdout_split,
    drop_leaky_fields,
    time_split,
)
from fx_credit.data.synthetic import generate_sovereign_panel


@pytest.fixture(scope="module")
def panel():
    return generate_sovereign_panel(seed=42)


def _tiny_panel():
    return pd.DataFrame(
        {
            "country": ["A", "A", "B", "B"],
            "year": [2000, 2001, 2000, 2001],
            "reserves_import_cover": [4.0, -1.0, 2.0, 3.0],
            "default": [0, 1, 0, 0],
        }
    )


def test_clean_removes_duplicates():
    df = _tiny_panel()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    out = clean_panel(dup)
    assert len(out) == len(df)
    assert not out.duplicated(subset=["country", "year"]).any()


def test_clean_nulls_impossible_values():
    out = clean_panel(_tiny_panel())
    row = out[(out.country == "A") & (out.year == 2001)]
    assert np.isnan(row["reserves_import_cover"].iloc[0])  # negative cover -> NaN


def test_clean_replaces_inf():
    df = _tiny_panel()
    df.loc[0, "reserves_import_cover"] = np.inf
    out = clean_panel(df)
    assert not np.isinf(out["reserves_import_cover"]).any()


def test_clean_sorted_by_country_year():
    df = _tiny_panel().sample(frac=1.0, random_state=1)
    out = clean_panel(df)
    assert out.equals(out.sort_values(["country", "year"]).reset_index(drop=True))


def test_clean_missing_columns_raises():
    with pytest.raises(ValueError, match="required columns"):
        clean_panel(pd.DataFrame({"country": ["A"], "year": [2000]}))


def test_clean_nonbinary_outcome_raises():
    df = _tiny_panel()
    df["default"] = [0, 2, 0, 1]
    with pytest.raises(ValueError, match="binary"):
        clean_panel(df)


def test_time_split_integrity(panel):
    train, test = time_split(panel, 2014)
    assert train["year"].max() == 2014
    assert test["year"].min() == 2015
    assert train["year"].max() < test["year"].min()


def test_time_split_no_same_country_future_row_in_train(panel):
    """A same-country future row must NEVER land in train under a time split."""
    train, test = time_split(panel, 2014)
    test_years = test.groupby("country")["year"].min()
    for country, first_test_year in test_years.items():
        c_train = train[train["country"] == country]
        assert (c_train["year"] < first_test_year).all()


def test_time_split_overlap_raises(panel):
    with pytest.raises(ValueError, match="leak"):
        time_split(panel, 2014, test_start_year=2014)


def test_time_split_gap_allowed(panel):
    train, test = time_split(panel, 2010, test_start_year=2015)
    assert train["year"].max() == 2010
    assert test["year"].min() == 2015


def test_leak_guard_raises_on_leaky_fields(panel):
    with pytest.raises(ValueError, match="leaky"):
        assert_no_leaky_fields(panel)


def test_leak_guard_passes_after_drop(panel):
    safe = drop_leaky_fields(panel)
    assert_no_leaky_fields(safe)  # should not raise
    for col in LEAKY_FIELDS:
        assert col not in safe.columns
    assert "reserves_import_cover" in safe.columns


def test_drop_leaky_extra_fields(panel):
    safe = drop_leaky_fields(panel, extra=("true_pd",))
    assert "true_pd" not in safe.columns


def test_country_holdout_disjoint(panel):
    train, test = country_holdout_split(panel, holdout_frac=0.2, seed=0)
    assert set(train["country"]) & set(test["country"]) == set()
    assert len(train) + len(test) == len(panel)
    n_hold = test["country"].nunique()
    assert n_hold == round(0.2 * panel["country"].nunique())


def test_country_holdout_deterministic(panel):
    _, t1 = country_holdout_split(panel, seed=5)
    _, t2 = country_holdout_split(panel, seed=5)
    assert set(t1["country"]) == set(t2["country"])


def test_country_holdout_invalid_frac(panel):
    with pytest.raises(ValueError, match="holdout_frac"):
        country_holdout_split(panel, holdout_frac=1.5)
