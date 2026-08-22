"""Variance-covariance VaR: closed forms, EWMA, Student-t, Cornish-Fisher."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from fx_var import (
    Book,
    PegBlindnessWarning,
    Spot,
    cornish_fisher_domain_ok,
    cornish_fisher_var,
    cornish_fisher_z,
    ewma_cov,
    normal_es,
    parametric_var,
    sample_cov,
    t_var,
    var_covar,
)
from fx_var.data.synthetic import demo_market


@pytest.fixture()
def market():
    return demo_market()


def test_var_covar_closed_form_hand_numbers():
    """w=[1,1], Sigma=diag(0.04,0.09): sigma_p = sqrt(0.13)."""
    w = pd.Series({"a": 1.0, "b": 1.0})
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["a", "b"], columns=["a", "b"])
    var, es = var_covar(w, cov, alpha=0.99)
    sig = np.sqrt(0.13)
    assert var == pytest.approx(sig * norm.ppf(0.99), rel=1e-12)
    assert es == pytest.approx(normal_es(sig, 0.99), rel=1e-12)


def test_var_covar_correlation_term():
    """Full quadratic form with off-diagonals, hand-expanded."""
    w = pd.Series({"a": 2.0, "b": -1.0})
    cov = pd.DataFrame([[0.04, 0.015], [0.015, 0.09]], index=["a", "b"], columns=["a", "b"])
    var, _ = var_covar(w, cov, alpha=0.95)
    sig = np.sqrt(4 * 0.04 + 1 * 0.09 + 2 * 2 * (-1) * 0.015)
    assert var == pytest.approx(sig * norm.ppf(0.95), rel=1e-12)


def test_parametric_var_matches_manual_single_factor(market):
    """Engine (FD exposures + sample cov) == hand formula N*S*sigma_r*z."""
    n = 10e6
    book = Book([Spot("EURUSD", n)])
    r = np.random.default_rng(8).normal(0, 0.006, 750)
    rets = pd.DataFrame({"FX:EUR": r})
    res = parametric_var(book, market, rets, alpha=0.99)
    manual = n * 1.08 * r.std(ddof=1) * norm.ppf(0.99)
    assert res.var == pytest.approx(manual, rel=1e-6)
    assert res.sigma == pytest.approx(n * 1.08 * r.std(ddof=1), rel=1e-6)


def test_horizon_scaling_sqrt_time(market):
    book = Book([Spot("EURUSD", 1e6)])
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.006, 300)})
    v1 = parametric_var(book, market, rets, 0.99, horizon_days=1)
    v10 = parametric_var(book, market, rets, 0.99, horizon_days=10)
    assert v10.var == pytest.approx(np.sqrt(10) * v1.var, rel=1e-9)


def test_ewma_cov_hand_recursion():
    r = pd.DataFrame({"x": [0.01, -0.02]})
    lam = 0.94
    s0 = float(r["x"].var(ddof=1))
    s1 = lam * s0 + 0.06 * 0.01**2
    s2 = lam * s1 + 0.06 * 0.02**2
    got = ewma_cov(r, lam).loc["x", "x"]
    assert got == pytest.approx(s2, rel=1e-12)


def test_ewma_cov_reacts_to_recent_vol(market):
    """EWMA covariance after a fresh vol spike exceeds the sample cov."""
    rng = np.random.default_rng(9)
    calm = rng.normal(0, 0.003, 480)
    spike = rng.normal(0, 0.02, 20)
    rets = pd.DataFrame({"FX:EUR": np.concatenate([calm, spike])})
    assert ewma_cov(rets).iloc[0, 0] > sample_cov(rets).iloc[0, 0]


def test_t_var_above_normal_at_99(market):
    book = Book([Spot("USDTRY", -10e6)])
    rets = pd.DataFrame({"FX:TRY": np.random.default_rng(10).standard_t(4, 750) * 0.01})
    vn = parametric_var(book, market, rets, 0.99, dist="normal")
    vt = parametric_var(book, market, rets, 0.99, dist="t", df=4)
    assert vt.var > vn.var
    assert vt.es > vn.es


def test_var_covar_invalid_dist():
    w = pd.Series({"a": 1.0})
    cov = pd.DataFrame([[0.01]], index=["a"], columns=["a"])
    with pytest.raises(ValueError, match="dist"):
        var_covar(w, cov, dist="cauchy")


def test_parametric_peg_warning(market):
    book = Book([Spot("USDHKD", -50e6)])
    rets = pd.DataFrame({"FX:HKD": np.random.default_rng(3).normal(0, 1e-4, 500)})
    with pytest.warns(PegBlindnessWarning):
        res = parametric_var(book, market, rets, 0.99)
    assert res.flagged_peg_factors == ("FX:HKD",)
    # and the VaR it reports is indeed near-nothing vs the ~$50m of HKD
    # exposure (< 0.1% of notional): that is the blindness
    assert res.var < 0.001 * 50e6


def test_parametric_invalid_cov_method(market):
    book = Book([Spot("EURUSD", 1e6)])
    rets = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.005, 300)})
    with pytest.raises(ValueError, match="cov_method"):
        parametric_var(book, market, rets, cov_method="shrinkage")


# ------------------------------------------------------------ Cornish-Fisher
def test_cf_reduces_to_normal_at_zero_moments():
    sigma, alpha = 3.2e5, 0.99
    cf = cornish_fisher_var(sigma, 0.0, 0.0, alpha)
    assert cf == pytest.approx(sigma * norm.ppf(alpha), rel=1e-12)


def test_cf_negative_skew_raises_left_tail_var():
    """Moderate carry-trade skew (still inside the CF validity domain)
    fattens the loss tail."""
    sigma, alpha = 1.0, 0.99
    base = cornish_fisher_var(sigma, 0.0, 0.0, alpha)
    skewed = cornish_fisher_var(sigma, -0.25, 0.0, alpha)
    assert skewed > base


def test_cf_excess_kurtosis_raises_extreme_tail_var():
    sigma = 1.0
    base = cornish_fisher_var(sigma, 0.0, 0.0, 0.99)
    fat = cornish_fisher_var(sigma, 0.0, 2.0, 0.99)
    assert fat > base


def test_cf_z_formula_hand_values():
    """z=2, S=-0.5, K=1: hand expansion of the CF polynomial."""
    z, s, k = 2.0, -0.5, 1.0
    expected = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    assert cornish_fisher_z(z, s, k) == pytest.approx(expected, rel=1e-14)


def test_cf_domain_check_rejects_extreme_moments():
    """Large |skew| breaks monotonicity: the engine must refuse."""
    assert cornish_fisher_domain_ok(0.0, 0.0)
    assert cornish_fisher_domain_ok(-0.5, 1.0)
    assert not cornish_fisher_domain_ok(-3.0, 0.0)
    with pytest.raises(ValueError, match="non-monotone"):
        cornish_fisher_var(1.0, -3.0, 0.0, 0.99)
    # explicit override still returns a number (documented escape hatch)
    forced = cornish_fisher_var(1.0, -3.0, 0.0, 0.99, check_domain=False)
    assert np.isfinite(forced)


def test_cf_invalid_sigma():
    with pytest.raises(ValueError):
        cornish_fisher_var(-1.0, 0.0, 0.0, 0.99)


def test_domain_check_is_exact_not_grid_resolution_dependent():
    # Regression for the closed-form rewrite of cornish_fisher_domain_ok.
    # On this module's default (z_range=4.0, n_grid=801), the previous
    # finite-difference-of-values grid check reported (skew, excess_kurt) =
    # (0.122, -0.427) as monotone -- every sampled z_cf value increased --
    # yet the true minimum of dz_cf/dz on [-4, 4] is ~ -9.15e-4 (near
    # z ~ 3.1, between two grid nodes), so the expansion is genuinely
    # non-monotone and the "quantile" it would produce is not a quantile.
    skew, excess_kurt = 0.122, -0.427
    assert not cornish_fisher_domain_ok(skew, excess_kurt)
    with pytest.raises(ValueError, match="non-monotone"):
        cornish_fisher_var(1.0, skew, excess_kurt, 0.99)
