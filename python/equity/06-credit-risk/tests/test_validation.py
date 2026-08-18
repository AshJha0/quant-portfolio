"""Tests for validation metrics: AUC vs sklearn, Gini/KS identities, HL
calibration, PSI exactness and drift detection, rank ordering, bootstrap."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eq_credit.data.synthetic import generate_loan_book
from eq_credit.validation import (
    bootstrap_auc_ci,
    brier_score,
    calibration_table,
    decile_table,
    gini,
    hosmer_lemeshow,
    is_monotone,
    ks_statistic,
    ks_table,
    psi,
    psi_from_proportions,
    psi_report,
    psi_status,
    roc_auc,
)


def test_auc_matches_sklearn_to_1e12() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    for seed in range(5):
        r = np.random.default_rng(seed)
        y = (r.uniform(size=500) < 0.3).astype(float)
        s = r.standard_normal(500)
        assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_with_ties_matches_sklearn() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(7)
    y = (rng.uniform(size=1_000) < 0.2).astype(float)
    s = rng.integers(0, 5, 1_000).astype(float)  # heavy ties
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_perfect_and_random() -> None:
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_true_pd_auc_near_theoretical_max() -> None:
    # The true-PD model achieves the maximum attainable AUC; a large sample
    # estimate of it should be stable (loose check around the known value).
    df = generate_loan_book(150_000, seed=17, missing=False, outliers=False)
    auc = roc_auc(df["default"].to_numpy(), df["true_pd"].to_numpy())
    assert 0.74 < auc < 0.84


def test_gini_identity() -> None:
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=800) < 0.25).astype(float)
    s = rng.standard_normal(800)
    assert gini(y, s) == pytest.approx(2 * roc_auc(y, s) - 1, abs=1e-14)


def test_ks_between_zero_and_one_and_perfect() -> None:
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    rng = np.random.default_rng(4)
    y2 = (rng.uniform(size=1_000) < 0.3).astype(float)
    s2 = rng.standard_normal(1_000)
    assert 0 <= ks_statistic(y2, s2) < 0.15  # random score: small KS


def test_ks_table_structure_and_max() -> None:
    df = generate_loan_book(20_000, seed=19)
    score = -df["true_pd"].to_numpy()  # higher = safer
    y = df["default"].to_numpy()
    tab = ks_table(y, score)
    assert len(tab) == 10
    assert tab["n"].sum() == len(df)
    # Decile 1 (lowest scores = riskiest) has the highest bad rate.
    assert tab["bad_rate"].iloc[0] == tab["bad_rate"].max()
    # Table KS is close to (and bounded by) the exact KS.
    exact = ks_statistic(y, -score)
    assert tab["ks"].max() <= exact + 1e-12
    assert tab["ks"].max() > 0.5 * exact


def test_bootstrap_auc_ci_contains_point_and_seeded() -> None:
    rng = np.random.default_rng(5)
    y = (rng.uniform(size=600) < 0.3).astype(float)
    s = y + rng.standard_normal(600)
    auc, lo, hi = bootstrap_auc_ci(y, s, n_boot=100, seed=11)
    assert lo <= auc <= hi
    assert (auc, lo, hi) == bootstrap_auc_ci(y, s, n_boot=100, seed=11)


def test_brier_score_hand_computed() -> None:
    y = np.array([1.0, 0.0])
    p = np.array([0.8, 0.3])
    assert brier_score(y, p) == pytest.approx((0.04 + 0.09) / 2, abs=1e-15)


def test_hosmer_lemeshow_null_distribution() -> None:
    # With KNOWN true probabilities (no estimated parameters) the HL statistic
    # is ~ chi2(df = 10 groups), mean 10.  Replication test, loose bounds
    # (mean of 60 chi2(10) draws has sd ~ 0.58).
    rng = np.random.default_rng(6)
    stats_ = []
    for _ in range(60):
        p = rng.uniform(0.01, 0.3, 2_000)
        y = (rng.uniform(size=2_000) < p).astype(float)
        chi2, _, _ = hosmer_lemeshow(y, p)
        stats_.append(chi2)
    assert 8.0 < np.mean(stats_) < 12.5


def test_hosmer_lemeshow_detects_miscalibration() -> None:
    rng = np.random.default_rng(7)
    p = rng.uniform(0.01, 0.3, 5_000)
    y = (rng.uniform(size=5_000) < np.clip(2.0 * p, 0, 1)).astype(float)
    chi2, pval, _ = hosmer_lemeshow(y, p)  # predictions half the true PD
    assert pval < 1e-6


def test_hl_table_and_calibration_table() -> None:
    rng = np.random.default_rng(8)
    p = rng.uniform(0.01, 0.2, 3_000)
    y = (rng.uniform(size=3_000) < p).astype(float)
    _, _, tab = hosmer_lemeshow(y, p)
    assert len(tab) == 10
    assert tab["n"].sum() == 3_000
    cal = calibration_table(y, p)
    assert {"predicted_pd", "observed_rate"} <= set(cal.columns)


# ------------------------------------------------------------------------ PSI
def test_psi_zero_for_identical_distributions() -> None:
    x = np.random.default_rng(9).standard_normal(5_000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-12)


def test_psi_hand_computed_exact_on_known_proportions() -> None:
    pe = np.array([0.5, 0.3, 0.2])
    pa = np.array([0.4, 0.4, 0.2])
    expected = (0.4 - 0.5) * np.log(0.4 / 0.5) + (0.4 - 0.3) * np.log(0.4 / 0.3)
    assert psi_from_proportions(pe, pa) == pytest.approx(expected, abs=1e-14)


def test_psi_detects_planted_shift() -> None:
    rng = np.random.default_rng(10)
    a = rng.standard_normal(10_000)
    b = rng.standard_normal(10_000) + 0.8  # large location shift
    v = psi(a, b)
    assert v > 0.25
    assert psi_status(v) == "shifted"


def test_psi_small_shift_monitor_band() -> None:
    rng = np.random.default_rng(11)
    a = rng.standard_normal(50_000)
    b = rng.standard_normal(50_000) + 0.4
    v = psi(a, b)
    assert 0.10 <= v <= 0.25
    assert psi_status(v) == "monitor"
    assert psi_status(0.01) == "stable"


def test_psi_proportions_validation() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        psi_from_proportions(np.array([0.5, 0.4]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="shape"):
        psi_from_proportions(np.array([1.0]), np.array([0.5, 0.5]))


def test_psi_handles_nan_bins() -> None:
    rng = np.random.default_rng(12)
    a = rng.standard_normal(5_000)
    b = rng.standard_normal(5_000)
    b[:2_000] = np.nan  # 40% missing in actual vs 0% expected
    assert psi(a, b) > 0.25


def test_psi_report_flags_drifted_feature() -> None:
    rng = np.random.default_rng(13)
    train = pd.DataFrame(
        {"stable": rng.standard_normal(8_000), "drifted": rng.standard_normal(8_000)}
    )
    oot = pd.DataFrame(
        {
            "stable": rng.standard_normal(8_000),
            "drifted": rng.standard_normal(8_000) + 1.0,
        }
    )
    rep = psi_report(train, oot, ["stable", "drifted"]).set_index("feature")
    assert rep.loc["drifted", "status"] == "shifted"
    assert rep.loc["stable", "status"] == "stable"


# --------------------------------------------------------------- rank ordering
def test_decile_default_rates_monotone_for_good_model() -> None:
    df = generate_loan_book(40_000, seed=23)
    tab = decile_table(df["default"].to_numpy(), df["true_pd"].to_numpy())
    assert len(tab) == 10
    # Small-sample noise tolerance of 20bp per step.
    assert is_monotone(tab["default_rate"].to_numpy(), increasing=True, tol=0.002)
    assert tab["default_rate"].iloc[-1] > 5 * max(tab["default_rate"].iloc[0], 1e-4)


def test_is_monotone_helper() -> None:
    assert is_monotone([1, 2, 3])
    assert not is_monotone([1, 3, 2])
    assert is_monotone([3, 2, 1], increasing=False)
    assert is_monotone([1.0, 0.999, 2.0], increasing=True, tol=0.01)


def test_single_class_target_raises() -> None:
    with pytest.raises(ValueError, match="single class"):
        roc_auc(np.zeros(10), np.arange(10.0))
