"""Cross-engine conventions (alpha, Kupiec, Basel, Cholesky, EWMA, MC).

Each test here failed on fx_var 1.0.0 and pins one rule of the shared
convention spec: alpha is the tail probability, Kupiec and Basel take
keyword counts, Basel works off the 250-day window, Cholesky jitter is
capped with explicit repair, PSD checks are relative, jumps scale with
the horizon, a negative shocked vol never aborts a Monte Carlo run, and the
bootstrap never builds an ``n_boot x n`` index matrix.
"""

from __future__ import annotations

import warnings

import fx_var as fv
import numpy as np
import pandas as pd
import pytest
from fx_var import (
    JumpSpec,
    NumericalWarning,
    basel_traffic_light,
    empirical_es,
    empirical_var,
    kupiec_pof,
    monte_carlo_var,
    nearest_correlation_higham,
    normal_es,
    normal_var,
    robust_cholesky,
    simulate_factor_returns,
    var_covar,
    var_standard_error,
    var_standard_error_bootstrap,
)
from fx_var.data.synthetic import demo_market
from scipy.stats import binom, norm

PNL = np.random.default_rng(0).standard_normal(400) * 1e5


# ------------------------------------------------------------------ S1
OLD_POSITIONAL = {
    "empirical_var": lambda a: fv.empirical_var(PNL, a),
    "empirical_es": lambda a: fv.empirical_es(PNL, a),
    "empirical_var_es": lambda a: fv.empirical_var_es(PNL, a),
    "normal_var": lambda a: fv.normal_var(1.0, a),
    "normal_es": lambda a: fv.normal_es(1.0, a),
    "t_var": lambda a: fv.t_var(1.0, a),
    "t_es": lambda a: fv.t_es(1.0, a),
    "var_covar": lambda a: var_covar(pd.Series({"x": 1.0}),
                                     pd.DataFrame([[1e-4]], index=["x"], columns=["x"]), a),
    "cornish_fisher_var": lambda a: fv.cornish_fisher_var(1.0, 0.0, 0.0, a),
    "var_standard_error": lambda a: var_standard_error(PNL, a),
    "var_standard_error_bootstrap": lambda a: var_standard_error_bootstrap(PNL, a, n_boot=5),
    "evaluate_var_backtest": lambda a: fv.evaluate_var_backtest(PNL[:50], np.full(50, 1e5), a),
    "conditional_coverage": lambda a: fv.conditional_coverage([0, 1, 0, 0], a),
}


@pytest.mark.parametrize("name", sorted(OLD_POSITIONAL))
def test_old_confidence_level_positional_raises(name):
    with pytest.raises(ValueError, match="alpha is the tail probability"):
        OLD_POSITIONAL[name](0.99)


@pytest.mark.parametrize("name", sorted(OLD_POSITIONAL))
def test_tail_alpha_accepted(name):
    out = OLD_POSITIONAL[name](0.01)
    assert out is not None


def test_confidence_keyword_is_deprecated_and_maps_to_one_minus():
    with pytest.warns(DeprecationWarning, match="confidence"):
        old = empirical_var(PNL, confidence=0.99)
    assert old == empirical_var(PNL, 0.01)
    with pytest.warns(DeprecationWarning):
        assert normal_es(2.0, confidence=0.975) == pytest.approx(normal_es(2.0, 0.025), rel=1e-12)
    with pytest.raises(ValueError, match="not both"):
        empirical_es(PNL, 0.01, confidence=0.99)


def test_tail_alpha_formulas():
    assert normal_var(1.0, 0.01) == pytest.approx(norm.ppf(0.99), rel=1e-14)
    assert normal_es(1.0, 0.025) == pytest.approx(norm.pdf(norm.ppf(0.975)) / 0.025, rel=1e-14)


def test_order_statistic_with_fractional_k():
    # n = 250, alpha = 0.01: k = 2.5, m = 3.  VaR is the 3rd worst loss, ES
    # weights the 3rd worst by 0.5.
    pnl = -np.arange(1.0, 251.0)  # losses 1..250
    assert empirical_var(pnl, 0.01) == 248.0
    assert empirical_es(pnl, 0.01) == pytest.approx((250 + 249 + 0.5 * 248) / 2.5, rel=1e-15)


# ------------------------------------------------------------------ S3
def test_kupiec_keyword_api_and_old_order():
    lr, _p = kupiec_pof(n_obs=250, n_exceptions=5, alpha=0.01)
    with pytest.warns(DeprecationWarning, match="positional"):
        lr_old, _p_old = kupiec_pof(5, 250, 0.99)
    assert lr_old == pytest.approx(lr, rel=1e-12)
    with pytest.raises(ValueError, match="must not exceed"):
        kupiec_pof(n_obs=10, n_exceptions=11, alpha=0.01)
    with pytest.raises(ValueError, match="tail probability"):
        kupiec_pof(n_obs=250, n_exceptions=5, alpha=0.99)
    with pytest.raises(ValueError):
        kupiec_pof(n_obs=0, n_exceptions=0, alpha=0.01)
    with pytest.raises(TypeError):
        kupiec_pof(n_obs=250, n_exceptions=5)


# ------------------------------------------------------------------ S4
@pytest.mark.parametrize("n, x, alpha, zone, mult", [
    (500, 9, 0.01, "yellow", 3.50),    # F = 0.9689, same as 250-day x = 6
    (500, 12, 0.01, "yellow", 3.75),   # F = 0.9981, 250-day x = 8
    (1000, 15, 0.01, "yellow", 3.40),  # F = 0.9521, 250-day x = 5
    (250, 12, 0.025, "yellow", 3.65),  # F = 0.9890, 250-day x = 7
    (500, 8, 0.01, "green", 3.00),
])
def test_basel_any_window_maps_to_250_day_equivalent(n, x, alpha, zone, mult):
    tl = basel_traffic_light(n_obs=n, n_exceptions=x, alpha=alpha)
    assert tl.zone == zone
    assert tl.multiplier == pytest.approx(mult, abs=1e-15)
    assert tl.plus_factor == pytest.approx(mult - 3.0, abs=1e-15)
    assert tl.cumulative_prob == pytest.approx(binom.cdf(x, n, alpha), abs=1e-12)


def test_basel_old_positional_form_deprecated():
    with pytest.warns(DeprecationWarning):
        tl = basel_traffic_light(5, 250, 0.99)
    assert (tl.zone, tl.multiplier) == ("yellow", 3.40)


# ------------------------------------------------------------------ S7
def test_cholesky_jitter_capped_at_1e_minus_6():
    # Smallest eigenvalue -5e-6 x mean variance: version 1.0.0 accepted it
    # with jitter 1e-5; the cap is now 1e-6 and the error names it.
    lam = np.array([1.0, 1.0, -5e-6])
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    a = (q * lam) @ q.T
    a = (a + a.T) / 2
    with pytest.raises(ValueError, match="most negative eigenvalue"):
        robust_cholesky(a)


def test_cholesky_small_jitter_warns():
    a = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular, PSD
    with pytest.warns(NumericalWarning, match="jitter"):
        robust_cholesky(a)


def test_cholesky_symmetry_is_relative():
    # Entries of order 1e-5 with an asymmetry of 1e-13 (1e-8 relative).
    a = np.array([[4e-5, 1e-5], [1e-5 + 1e-13, 3e-5]])
    with pytest.raises(ValueError, match="symmetric"):
        robust_cholesky(a)


def test_higham_and_clip_repair():
    a = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    with pytest.raises(ValueError):
        robust_cholesky(a)
    for method in ("clip", "higham"):
        with pytest.warns(NumericalWarning):
            chol = robust_cholesky(a, repair=method)
        b = chol @ chol.T
        np.testing.assert_allclose(np.diag(b), 1.0, atol=1e-6)
    c = nearest_correlation_higham(a)
    np.testing.assert_allclose(np.diag(c), 1.0, atol=0)
    assert np.linalg.eigvalsh(c)[0] > -1e-12
    # Higham is the nearest: closer to A in Frobenius norm than clip.
    from fx_var.monte_carlo_var import _repair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip = _repair(a, "clip")
    assert np.linalg.norm(c - a) <= np.linalg.norm(clip - a) + 1e-12


def test_max_tries_is_deprecated():
    with pytest.warns(DeprecationWarning):
        robust_cholesky(np.eye(2), max_tries=3)


# ------------------------------------------------------------------ H-9
def test_psd_check_relative_large_book_rounding_does_not_raise():
    # Rank-one PSD covariance, book orthogonal to it: w'Sw = 0 exactly in
    # real arithmetic, about -2e-8 in floating point at this scale.
    v = np.array([0.0006286511054669665, -0.0006605243164565094, 0.0032021132522164104])
    w = pd.Series([1542958.7726938147, -44166260.250844546, -9413430.82725425],
                  index=list("abc"))
    cov = pd.DataFrame(np.outer(v, v), index=list("abc"), columns=list("abc"))
    var, _ = var_covar(w, cov, 0.01)
    assert abs(var) < 1e-2


def test_psd_check_relative_small_book_indefinite_raises():
    # Indefinite covariance seen by a tiny book: w'Sw = -2e-14, which the old
    # absolute threshold of -1e-12 accepted as zero risk.
    cov = pd.DataFrame([[1e-4, 2e-4], [2e-4, 1e-4]], index=["a", "b"], columns=["a", "b"])
    w = pd.Series({"a": 1e-5, "b": -1e-5})
    with pytest.raises(ValueError, match="positive semi-definite"):
        var_covar(w, cov, 0.01)


# ------------------------------------------------------------------ S10
def test_jump_count_is_poisson_over_the_horizon():
    cov = pd.DataFrame([[1e-12]], index=["FX:TRY"], columns=["FX:TRY"])
    p, h, size = 0.05, 10.0, -0.10
    js = JumpSpec(prob=p, mean={"FX:TRY": size})
    x = simulate_factor_returns(cov, 400_000, "jump", jumps=js, seed=3, horizon_days=h)
    lam_h = -np.log1p(-p) * h
    se = abs(size) * np.sqrt(lam_h / 400_000)
    assert x["FX:TRY"].mean() == pytest.approx(size * lam_h, abs=5 * se)
    # P(no jump) = exp(-lam h) = (1 - p)^h
    assert np.mean(np.abs(x["FX:TRY"]) < 1e-3) == pytest.approx((1 - p) ** h, abs=3e-3)


def test_jump_prob_one_rejected():
    with pytest.raises(ValueError):
        JumpSpec(prob=1.0, mean={"FX:TRY": -0.1})


def test_mc_never_raises_on_a_negative_vol_path():
    # Version 1.0.0 raised "shocked implied vol ... is negative" here.
    mkt = demo_market()
    book = fv.Book([fv.Option("EURUSD", 10e6, 1.08, 0.25)])
    cov = pd.DataFrame(np.diag([6e-5, 1e-8, 1e-8, 4e-4]),
                       index=book.factors(mkt), columns=book.factors(mkt))
    with pytest.warns(NumericalWarning, match="floored"):
        res = monte_carlo_var(book, mkt, cov, 0.01, horizon_days=10,
                              n_scenarios=5000, dist="t", df=2.5, seed=0, n_boot=20)
    assert np.isfinite(res.var) and np.isfinite(res.es) and np.isfinite(res.se_var)


def test_bootstrap_memory_is_linear_in_n():
    # 200k scenarios x 50 resamples: an n_boot x n int64 index matrix is
    # 80 MB.  One resample at a time peaks near 3 arrays of n (5 MB).
    import tracemalloc

    x = np.random.default_rng(2).standard_normal(200_000)
    tracemalloc.start()
    try:
        var_standard_error_bootstrap(x, 0.01, n_boot=50, seed=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 20e6
    with pytest.raises(ValueError, match="n_boot"):
        var_standard_error_bootstrap(PNL, 0.01, n_boot=1)


def test_kde_se_uses_silverman_at_the_reported_quantile():
    x = np.random.default_rng(5).standard_normal(1000)
    alpha = 0.01
    q = -empirical_var(x, alpha)
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    h = 0.9 * min(sd, iqr / 1.34) * 1000 ** (-0.2)
    dens = np.mean(norm.pdf((q - x) / h)) / h
    want = np.sqrt(alpha * (1 - alpha) / 1000) / dens
    assert var_standard_error(x, alpha) == pytest.approx(want, rel=1e-12)


def test_mc_se_is_the_bootstrap_of_the_same_scenarios():
    mkt = demo_market()
    book = fv.Book([fv.Spot("EURUSD", 10e6)])
    cov = pd.DataFrame([[3.6e-5]], index=["FX:EUR"], columns=["FX:EUR"])
    res = monte_carlo_var(book, mkt, cov, 0.01, n_scenarios=20_000, seed=4, n_boot=100)
    exact = normal_var(10e6 * 1.08 * 6e-3, 0.01)
    assert abs(res.var - exact) < 5 * res.se_var
    assert 0 < res.se_var < 0.05 * res.var
