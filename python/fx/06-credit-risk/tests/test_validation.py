"""AUC/Gini/KS/HL/PSI from scratch: hand checks + sklearn/scipy cross-checks."""

import warnings

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fx_credit.data.synthetic import generate_sovereign_panel
from fx_credit.validation import (
    auc,
    bootstrap_auc_ci,
    gini,
    hosmer_lemeshow,
    ks_statistic,
    psi,
    within_country_autocorrelation,
)


def test_auc_matches_sklearn_continuous():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000).astype(float)
    s = rng.standard_normal(4000) + 0.7 * y
    assert abs(auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000).astype(float)
    s = rng.integers(0, 6, 4000).astype(float)  # heavy ties
    assert abs(auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_auc_perfect_and_reversed():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auc_hand_computed():
    # pairs: (b=0.35 vs g=0.1: win), (0.35 vs 0.4: loss), (0.8 vs both: 2 wins)
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert auc(y, s) == pytest.approx(3.0 / 4.0, abs=1e-14)


def test_gini_identity():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500).astype(float)
    s = rng.standard_normal(500)
    assert gini(y, s) == pytest.approx(2.0 * auc(y, s) - 1.0, abs=1e-14)


def test_auc_single_class_raises():
    with pytest.raises(ValueError, match="both classes"):
        auc(np.zeros(10), np.arange(10.0))


def test_ks_hand_computed():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    # sorted: 0.1(g) 0.35(b) 0.4(g) 0.8(b); |F_b - F_g| = .5, 0, .5, 0
    assert ks_statistic(y, s) == pytest.approx(0.5, abs=1e-14)


def test_ks_perfect_separation_is_one():
    y = np.r_[np.zeros(50), np.ones(50)]
    s = np.r_[np.linspace(0, 1, 50), np.linspace(2, 3, 50)]
    assert ks_statistic(y, s) == pytest.approx(1.0, abs=1e-14)


def test_ks_bounds():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300).astype(float)
    s = rng.standard_normal(300)
    assert 0.0 <= ks_statistic(y, s) <= 1.0


def test_hosmer_lemeshow_hand_computed():
    """3 groups of 2, chi2 = sum (O-E)^2 / (E (1 - E/n))."""
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7])
    res = hosmer_lemeshow(y, p, n_groups=3)
    # sorted p groups: (.1,.2) O=0 E=.3; (.3,.4) O=1 E=.7; (.6,.7) O=2 E=1.3
    exp = 0.0
    for o, e, n in [(0, 0.3, 2), (1, 0.7, 2), (2, 1.3, 2)]:
        exp += (o - e) ** 2 / (e * (1 - e / n))
    assert res.chi2 == pytest.approx(exp, abs=1e-12)
    assert res.dof == 1


def test_hosmer_lemeshow_well_calibrated():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.3, 20_000)
    y = (rng.random(20_000) < p).astype(float)
    res = hosmer_lemeshow(y, p)
    assert res.p_value > 0.01  # no evidence of miscalibration


def test_hosmer_lemeshow_detects_miscalibration():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.01, 0.3, 20_000)
    y = (rng.random(20_000) < p).astype(float)
    res = hosmer_lemeshow(y, p / 2.0)  # PDs understated 2x
    assert res.p_value < 1e-6 and res.chi2 > 100


def test_hosmer_lemeshow_invalid_groups():
    with pytest.raises(ValueError, match="n_groups"):
        hosmer_lemeshow(np.array([0.0, 1.0]), np.array([0.1, 0.2]), n_groups=2)


def test_psi_identical_is_zero():
    x = np.linspace(0, 1, 1000)
    assert psi(x, x.copy()) == pytest.approx(0.0, abs=1e-12)


def test_psi_hand_computed():
    """Explicit 2-bin case: expected shares (.5,.5), actual (.8,.2).

    PSI = (.8-.5) ln(.8/.5) + (.2-.5) ln(.2/.5) = .3 ln1.6 + .3 ln2.5.
    """
    edges = np.array([0.0, 1.0, 2.0])
    e = np.r_[np.full(50, 0.5), np.full(50, 1.5)]
    a = np.r_[np.full(80, 0.5), np.full(20, 1.5)]
    expected = 0.3 * np.log(1.6) + 0.3 * np.log(2.5)
    assert psi(e, a, edges=edges) == pytest.approx(expected, abs=1e-12)


def test_psi_flags_large_shift():
    rng = np.random.default_rng(6)
    e = rng.standard_normal(5000)
    a = rng.standard_normal(5000) + 1.5
    assert psi(e, a) > 0.25


def test_psi_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        psi(np.array([]), np.array([1.0]))


def test_bootstrap_auc_ci_brackets_point():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 800).astype(float)
    s = rng.standard_normal(800) + 0.8 * y
    point, lo, hi = bootstrap_auc_ci(y, s, n_boot=200, seed=1)
    assert lo <= point <= hi
    assert hi - lo > 0.0
    assert bootstrap_auc_ci(y, s, n_boot=200, seed=1) == (point, lo, hi)  # seeded


def test_bootstrap_ci_wide_for_low_default():
    """Low-default portfolio: few events -> wide AUC CI (documented failure mode)."""
    rng = np.random.default_rng(8)
    n = 400
    p = np.clip(rng.uniform(0.0, 0.08, n), 1e-4, 1)
    y = (rng.random(n) < p).astype(float)
    if y.sum() < 5:
        y[:5] = 1.0
    point, lo, hi = bootstrap_auc_ci(y, p, n_boot=300, seed=2)
    assert hi - lo > 0.10


def test_within_country_autocorrelation_warns_on_ar1():
    df = generate_sovereign_panel(seed=42)
    with pytest.warns(UserWarning, match="autocorrelation"):
        ser = within_country_autocorrelation(df, "reserves_import_cover")
    assert ser.mean(skipna=True) > 0.3


def test_within_country_autocorrelation_silent_on_iid():
    rng = np.random.default_rng(9)
    df = generate_sovereign_panel(seed=42)[["country", "year"]].copy()
    df["noise"] = rng.standard_normal(len(df))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ser = within_country_autocorrelation(df, "noise")
    assert abs(ser.mean(skipna=True)) < 0.10


def test_within_country_autocorrelation_missing_col_raises():
    df = generate_sovereign_panel(seed=42)
    with pytest.raises(ValueError, match="not in frame"):
        within_country_autocorrelation(df, "nope")
