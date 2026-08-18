"""Tests for the cleaning pipeline: leakage guard, winsorization, duplicates,
missing-value strategies, train/OOT temporal split."""

import numpy as np
import pandas as pd
import pytest

from eq_credit.cleaning import (
    FORBIDDEN_POST_OUTCOME_FIELDS,
    LeakageError,
    MedianImputer,
    apply_winsor,
    check_leakage,
    drop_duplicate_loans,
    find_duplicates,
    fit_winsor_bounds,
    train_oot_split,
)
from eq_credit.data.synthetic import generate_loan_book


def test_leakage_guard_catches_planted_field() -> None:
    with pytest.raises(LeakageError, match="writeoff_flag"):
        check_leakage(["leverage", "writeoff_flag", "roa"])


def test_leakage_guard_lists_all_offenders() -> None:
    with pytest.raises(LeakageError) as ei:
        check_leakage(["recovery_amount", "days_past_due_max"])
    assert "recovery_amount" in str(ei.value)
    assert "days_past_due_max" in str(ei.value)


def test_leakage_guard_passes_clean_list() -> None:
    check_leakage(["leverage", "roa", "sector"])  # no raise


def test_forbidden_list_covers_generator_leaks() -> None:
    df = generate_loan_book(500, seed=1)
    planted = {"writeoff_flag", "recovery_amount", "days_past_due_max"}
    assert planted <= set(FORBIDDEN_POST_OUTCOME_FIELDS)
    assert planted <= set(df.columns)


def test_winsorization_bounds_and_apply() -> None:
    df = pd.DataFrame({"x": np.arange(1, 101, dtype=float)})
    b = fit_winsor_bounds(df, ["x"], lower_q=0.05, upper_q=0.95)
    out = apply_winsor(df, b)
    assert out["x"].min() == pytest.approx(df["x"].quantile(0.05))
    assert out["x"].max() == pytest.approx(df["x"].quantile(0.95))
    # Interior values untouched.
    mid = (df["x"] > b.lower["x"]) & (df["x"] < b.upper["x"])
    assert (out.loc[mid, "x"] == df.loc[mid, "x"]).all()


def test_winsorization_tames_planted_outliers() -> None:
    df = generate_loan_book(10_000, seed=11, outliers=True)
    b = fit_winsor_bounds(df, ["leverage", "current_ratio"])
    out = apply_winsor(df, b)
    assert out["leverage"].max() < df["leverage"].max()
    assert out["leverage"].max() <= b.upper["leverage"]


def test_winsor_preserves_nan() -> None:
    df = pd.DataFrame({"x": [1.0, np.nan, 100.0]})
    b = fit_winsor_bounds(df, ["x"], 0.10, 0.90)
    out = apply_winsor(df, b)
    assert np.isnan(out["x"].iloc[1])


def test_winsor_invalid_quantiles_raise() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="lower_q"):
        fit_winsor_bounds(df, ["x"], 0.9, 0.1)


def test_duplicate_detection_and_drop() -> None:
    df = generate_loan_book(1_000, seed=4, n_duplicates=12)
    dups = find_duplicates(df)
    assert len(dups) == 24  # both occurrences of the 12 duplicated ids
    clean = drop_duplicate_loans(df)
    assert len(clean) == 1_000
    assert not clean["loan_id"].duplicated().any()


def test_find_duplicates_bad_key_raises() -> None:
    with pytest.raises(ValueError, match="key column"):
        find_duplicates(pd.DataFrame({"a": [1]}), key="loan_id")


def test_median_imputer_fills_with_train_median() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan, 10.0]})
    imp = MedianImputer().fit(train, ["x"])
    out = imp.transform(test)
    assert out["x"].iloc[0] == 2.0  # train median, not test median
    assert out["x"].iloc[1] == 10.0


def test_median_imputer_unfitted_raises() -> None:
    with pytest.raises(ValueError, match="before fit"):
        MedianImputer().transform(pd.DataFrame({"x": [1.0]}))


def test_train_oot_split_temporal_integrity() -> None:
    df = generate_loan_book(5_000, seed=8, start="2019-01-01", end="2022-12-31")
    train, oot = train_oot_split(df, cutoff="2022-01-01")
    assert pd.to_datetime(train["origination_date"]).max() < pd.Timestamp("2022-01-01")
    assert pd.to_datetime(oot["origination_date"]).min() >= pd.Timestamp("2022-01-01")
    assert len(train) + len(oot) == len(df)


def test_train_oot_split_empty_side_raises() -> None:
    df = generate_loan_book(500, seed=9, start="2019-01-01", end="2019-12-31")
    with pytest.raises(ValueError, match="empty side"):
        train_oot_split(df, cutoff="2030-01-01")
