"""Tests for the synthetic loan-book generator: base rate, ground-truth
monotonicity, seeding, drift, missingness and planted pathologies."""

import numpy as np
import pandas as pd
import pytest

from eq_credit.data.synthetic import (
    SECTORS,
    TRUE_COEFFS,
    generate_loan_book,
    generate_oot_sample,
    true_log_odds,
)


@pytest.fixture(scope="module")
def book() -> pd.DataFrame:
    return generate_loan_book(30_000, seed=7)


def test_base_default_rate_near_target(book: pd.DataFrame) -> None:
    # Mean true PD is calibrated by bisection; realised rate is binomial noise.
    assert abs(book["true_pd"].mean() - 0.03) < 1e-3
    assert abs(book["default"].mean() - 0.03) < 0.005


def test_custom_target_rate() -> None:
    df = generate_loan_book(20_000, seed=1, target_default_rate=0.05)
    assert abs(df["true_pd"].mean() - 0.05) < 1e-3


def test_true_pd_monotone_in_leverage() -> None:
    # Hold everything else fixed; vary leverage below the cap.
    lev = np.linspace(0.05, 0.95, 20)
    n = len(lev)
    eta = true_log_odds(
        lev,
        np.full(n, 3.0),
        np.full(n, 1.5),
        np.full(n, 0.05),
        np.full(n, 17.0),
        np.full(n, 60.0),
        np.array(["manufacturing"] * n),
    )
    assert np.all(np.diff(eta) > 0)


def test_leverage_effect_capped_above_one() -> None:
    n = 2
    base = dict(
        interest_coverage=np.full(n, 3.0), current_ratio=np.full(n, 1.5),
        roa=np.full(n, 0.05), log_assets=np.full(n, 17.0),
        behavioral_score=np.full(n, 60.0), sector=np.array(["services"] * n),
    )
    eta = true_log_odds(np.array([1.0, 1.8]), **base)
    assert eta[0] == pytest.approx(eta[1])  # flat beyond leverage = 1


def test_current_ratio_u_shape() -> None:
    n = 3
    base = dict(
        leverage=np.full(n, 0.4), interest_coverage=np.full(n, 3.0),
        roa=np.full(n, 0.05), log_assets=np.full(n, 17.0),
        behavioral_score=np.full(n, 60.0), sector=np.array(["services"] * n),
    )
    eta = true_log_odds(current_ratio=np.array([0.3, 1.5, 8.0]), **base)
    assert eta[0] > eta[1] and eta[2] > eta[1]  # both extremes riskier


def test_noise_features_have_zero_true_effect() -> None:
    assert TRUE_COEFFS["noise_1"] == 0.0
    assert TRUE_COEFFS["noise_2"] == 0.0


def test_seeded_reproducibility() -> None:
    a = generate_loan_book(2_000, seed=99)
    b = generate_loan_book(2_000, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ() -> None:
    a = generate_loan_book(2_000, seed=1)
    b = generate_loan_book(2_000, seed=2)
    assert not a["leverage"].equals(b["leverage"])


def test_missingness_patterns(book: pd.DataFrame) -> None:
    # MCAR on current_ratio ~4%; informative on behavioral_score.
    assert 0.02 < book["current_ratio"].isna().mean() < 0.06
    miss = book["behavioral_score"].isna()
    assert miss.mean() > 0.03
    # Informative: missing rows have higher true PD than observed rows.
    assert book.loc[miss, "true_pd"].mean() > 1.3 * book.loc[~miss, "true_pd"].mean()


def test_post_outcome_fields_are_leaky(book: pd.DataFrame) -> None:
    d = book["default"]
    assert (book.loc[d == 0, "recovery_amount"] == 0).all()
    assert (book["writeoff_flag"] == d).mean() > 0.9
    assert book.loc[d == 1, "days_past_due_max"].min() >= 90


def test_duplicates_injected() -> None:
    df = generate_loan_book(1_000, seed=3, n_duplicates=17)
    assert len(df) == 1_017
    assert df["loan_id"].duplicated().sum() == 17


def test_oot_sample_dates_and_drift() -> None:
    train = generate_loan_book(10_000, seed=5)
    oot = generate_oot_sample(10_000, seed=6, drift=0.6, calibration_shift=0.3)
    assert pd.to_datetime(oot["origination_date"]).min() > pd.to_datetime(
        train["origination_date"]
    ).max()
    # Drift shifts leverage up; calibration shift raises the realised base rate.
    assert oot["leverage"].mean() > train["leverage"].mean() + 0.03
    assert oot["true_pd"].mean() > train["true_pd"].mean() * 1.1


def test_invalid_n_raises() -> None:
    with pytest.raises(ValueError, match="n_loans"):
        generate_loan_book(0, seed=1)


def test_sector_labels_valid(book: pd.DataFrame) -> None:
    assert set(book["sector"].unique()) <= set(SECTORS)
