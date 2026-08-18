"""Tests for WOE/IV: hand-computed exact values, monotone merging, missing
bin, zero-cell smoothing, leakage flag on planted feature, noise IV."""

import warnings

import numpy as np
import pandas as pd
import pytest

from eq_credit.data.synthetic import generate_loan_book
from eq_credit.validation import is_monotone
from eq_credit.woe import (
    SuspiciousIVWarning,
    WOETransformer,
    fit_categorical_binning,
    fit_numeric_binning,
    iv_strength,
    woe_iv_from_counts,
)


# ---------------------------------------------------------------- hand checks
def test_woe_hand_computed_exact() -> None:
    # Known tiny table: goods per bin, bads per bin (no zero cells).
    good = np.array([90.0, 60.0, 30.0])   # total 180
    bad = np.array([5.0, 15.0, 25.0])     # total 45
    woe, ivc, iv = woe_iv_from_counts(good, bad)
    # WOE_i = ln( (good_i/180) / (bad_i/45) ), by hand:
    expected_woe = np.log((good / 180.0) / (bad / 45.0))
    np.testing.assert_allclose(woe, expected_woe, rtol=0, atol=1e-14)
    # e.g. bin 1: ln(0.5 / 0.111...) = ln(4.5)
    assert woe[0] == pytest.approx(np.log(4.5), abs=1e-14)


def test_iv_hand_computed_exact() -> None:
    good = np.array([90.0, 60.0, 30.0])
    bad = np.array([5.0, 15.0, 25.0])
    _, _, iv = woe_iv_from_counts(good, bad)
    dg, db = good / 180.0, bad / 45.0
    expected_iv = float(np.sum((dg - db) * np.log(dg / db)))
    assert iv == pytest.approx(expected_iv, abs=1e-14)


def test_woe_sign_convention() -> None:
    # A bin with more than its share of goods must have POSITIVE WOE.
    woe, _, _ = woe_iv_from_counts(np.array([90.0, 10.0]), np.array([10.0, 90.0]))
    assert woe[0] > 0 > woe[1]


def test_zero_cell_smoothing_no_inf() -> None:
    good = np.array([50.0, 50.0])
    bad = np.array([0.0, 10.0])  # zero-default bin
    woe, _, iv = woe_iv_from_counts(good, bad)
    assert np.isfinite(woe).all() and np.isfinite(iv)
    # Non-zero bin unaffected by the smoothing of the other bin.
    assert woe[1] == pytest.approx(np.log((50 / 100) / (10 / 10)), abs=1e-12)


def test_all_one_class_raises() -> None:
    with pytest.raises(ValueError, match="goods and bads"):
        woe_iv_from_counts(np.array([10.0, 10.0]), np.array([0.0, 0.0]))


def test_iv_strength_thresholds() -> None:
    assert iv_strength(0.01) == "useless"
    assert iv_strength(0.05) == "weak"
    assert iv_strength(0.2) == "medium"
    assert iv_strength(0.4) == "strong"
    assert iv_strength(0.9) == "suspicious"
    with pytest.raises(ValueError):
        iv_strength(-0.1)


# ------------------------------------------------------------ fitted binnings
@pytest.fixture(scope="module")
def book() -> pd.DataFrame:
    return generate_loan_book(25_000, seed=21)


def test_monotone_merging_produces_monotone_woe(book: pd.DataFrame) -> None:
    fb = fit_numeric_binning(book["leverage"], book["default"], "leverage")
    # WOE across non-missing bins must be monotone (decreasing for a risk
    # driver: higher leverage -> lower WOE).
    assert is_monotone(fb.woes, increasing=False)
    assert len(fb.woes) >= 2


def test_bad_rate_monotone_after_merge(book: pd.DataFrame) -> None:
    fb = fit_numeric_binning(book["interest_coverage"], book["default"], "ic")
    rates = fb.table.loc[fb.table["bin"] != "MISSING", "bad_rate"].to_numpy()
    assert is_monotone(rates, increasing=True) or is_monotone(rates, increasing=False)


def test_u_shape_preserved_without_monotone_constraint(book: pd.DataFrame) -> None:
    fb = fit_numeric_binning(
        book["current_ratio"], book["default"], "cr", monotone=False,
        min_bin_frac=0.08,
    )
    rates = fb.table.loc[fb.table["bin"] != "MISSING", "bad_rate"].to_numpy()
    # U-shape: both end bins riskier than the middle minimum.
    assert rates[0] > rates.min() and rates[-1] > rates.min()
    assert 0 < int(np.argmin(rates)) < len(rates) - 1


def test_missing_bin_is_separate(book: pd.DataFrame) -> None:
    fb = fit_numeric_binning(book["behavioral_score"], book["default"], "beh")
    assert fb.has_missing_bin
    assert "MISSING" in fb.table["bin"].values
    # Informative missingness: thin-file (missing) obligors are riskier than
    # average, so the missing bin's WOE is negative.
    assert fb.missing_woe < 0


def test_transform_maps_values_and_missing(book: pd.DataFrame) -> None:
    fb = fit_numeric_binning(book["behavioral_score"], book["default"], "beh")
    x = pd.Series([np.nan, 5.0, 95.0])
    out = fb.transform(x)
    assert out[0] == pytest.approx(fb.missing_woe)
    assert out[1] == pytest.approx(fb.woes[0])     # lowest bin
    assert out[2] == pytest.approx(fb.woes[-1])    # highest bin
    assert out[1] < out[2]  # low score risky, high score safe


def test_leaky_feature_triggers_suspicious_iv_warning(book: pd.DataFrame) -> None:
    with pytest.warns(SuspiciousIVWarning, match="writeoff_flag"):
        fb = fit_numeric_binning(
            book["writeoff_flag"], book["default"], "writeoff_flag", n_prebins=4
        )
    assert fb.iv > 0.5


def test_noise_feature_iv_below_useless_threshold(book: pd.DataFrame) -> None:
    fb = fit_numeric_binning(book["noise_1"], book["default"], "noise_1")
    assert fb.iv < 0.02


def test_real_features_iv_ordering(book: pd.DataFrame) -> None:
    fb_lev = fit_numeric_binning(book["leverage"], book["default"], "lev")
    fb_noise = fit_numeric_binning(book["noise_2"], book["default"], "n2")
    assert fb_lev.iv > 0.1 > fb_noise.iv


def test_zero_default_sample_raises(book: pd.DataFrame) -> None:
    x = book["leverage"].head(100)
    y = pd.Series(np.zeros(100, dtype=int))
    with pytest.raises(ValueError, match="zero defaults"):
        fit_numeric_binning(x, y, "lev")


def test_categorical_binning_sector(book: pd.DataFrame) -> None:
    fb = fit_categorical_binning(book["sector"], book["default"], "sector")
    assert fb.kind == "categorical"
    assert fb.iv > 0.01
    # Construction (highest true sector effect) should be riskier than
    # healthcare (lowest): lower WOE.
    tab = fb.table.set_index("bin")
    assert tab.loc["construction", "woe"] < tab.loc["healthcare", "woe"]


def test_categorical_transform_unseen_maps_to_missing_woe(book: pd.DataFrame) -> None:
    fb = fit_categorical_binning(book["sector"], book["default"], "sector")
    out = fb.transform(pd.Series(["martian_mining"]))
    assert out[0] == pytest.approx(fb.missing_woe)


def test_transformer_fit_transform_report(book: pd.DataFrame) -> None:
    wt = WOETransformer(["leverage", "roa"], ["sector"]).fit(book, "default")
    woe = wt.transform(book)
    assert set(woe.columns) == {"woe_leverage", "woe_roa", "woe_sector"}
    assert woe.notna().all().all()
    ivt = wt.iv_table()
    assert list(ivt.columns) == ["feature", "iv", "strength"]
    assert ivt["iv"].is_monotonic_decreasing


def test_transformer_before_fit_raises(book: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="before fit"):
        WOETransformer(["leverage"]).transform(book)


def test_binning_table_counts_consistent(book: pd.DataFrame) -> None:
    fb = fit_numeric_binning(book["leverage"], book["default"], "lev")
    assert fb.table["n"].sum() == len(book)
    assert fb.table["n_bad"].sum() == book["default"].sum()
    assert fb.iv == pytest.approx(fb.table["iv_contrib"].sum(), abs=1e-12)
