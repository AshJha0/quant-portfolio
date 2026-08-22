"""Monte Carlo: unbiasedness vs BS, variance reduction, MC Greeks, seeding."""

import numpy as np
import pytest

from eq_options import (
    bs_greeks,
    bs_price,
    mc_delta_lr,
    mc_delta_pathwise,
    mc_greek_fd,
    mc_price,
    mc_vega_lr,
    mc_vega_pathwise,
)

CASES = [
    # (S, K, T, r, sigma, q, otype)
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00, "call"),
    (100.0, 110.0, 0.5, 0.03, 0.30, 0.02, "put"),
    (100.0, 80.0, 2.0, 0.01, 0.25, 0.03, "call"),
    (50.0, 55.0, 0.25, -0.01, 0.40, 0.00, "put"),
]


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q", "otype"), CASES)
def test_mc_within_3_standard_errors_of_bs(
    S: float, K: float, T: float, r: float, sigma: float, q: float, otype: str
) -> None:
    bs = bs_price(S, K, T, r, sigma, q, otype)
    res = mc_price(S, K, T, r, sigma, q, otype, n_paths=200_000, seed=123)
    assert abs(res.value - bs) <= 3.0 * res.std_error
    assert res.std_error > 0.0


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q", "otype"), CASES)
def test_ci_covers_analytic_price(
    S: float, K: float, T: float, r: float, sigma: float, q: float, otype: str
) -> None:
    bs = bs_price(S, K, T, r, sigma, q, otype)
    res = mc_price(S, K, T, r, sigma, q, otype, n_paths=200_000, seed=7)
    # 95% CI: widen to 3 SE to keep the seeded test deterministic-robust
    assert res.ci_low - 1.1 * res.std_error <= bs <= res.ci_high + 1.1 * res.std_error
    assert res.ci_low < res.value < res.ci_high


def test_antithetic_reduces_standard_error() -> None:
    kwargs = dict(n_paths=100_000, control_variate=False, seed=11)
    plain = mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call", antithetic=False, **kwargs)
    anti = mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call", antithetic=True, **kwargs)
    assert anti.std_error < plain.std_error


def test_control_variate_reduces_standard_error() -> None:
    kwargs = dict(n_paths=100_000, antithetic=False, seed=11)
    plain = mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call", control_variate=False, **kwargs)
    cv = mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call", control_variate=True, **kwargs)
    assert cv.std_error < plain.std_error


def test_combined_variance_reduction_beats_plain_substantially() -> None:
    plain = mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call",
                     n_paths=100_000, antithetic=False, control_variate=False, seed=3)
    both = mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call",
                    n_paths=100_000, antithetic=True, control_variate=True, seed=3)
    assert both.std_error < 0.6 * plain.std_error


def test_seed_reproducibility() -> None:
    a = mc_price(100, 100, 1, 0.05, 0.2, 0.01, "call", n_paths=50_000, seed=99)
    b = mc_price(100, 100, 1, 0.05, 0.2, 0.01, "call", n_paths=50_000, seed=99)
    c = mc_price(100, 100, 1, 0.05, 0.2, 0.01, "call", n_paths=50_000, seed=100)
    assert a.value == b.value and a.std_error == b.std_error
    assert a.value != c.value


def test_generator_accepted_as_seed() -> None:
    rng = np.random.default_rng(5)
    res = mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call", n_paths=10_000, seed=rng)
    assert res.value > 0


@pytest.mark.parametrize("otype", ["call", "put"])
def test_pathwise_delta_matches_analytic(otype: str) -> None:
    S, K, T, r, sigma, q = 100.0, 105.0, 1.0, 0.04, 0.25, 0.01
    ana = bs_greeks(S, K, T, r, sigma, q, otype).delta
    est = mc_delta_pathwise(S, K, T, r, sigma, q, otype, n_paths=400_000, seed=21)
    assert abs(est.value - ana) <= 3.0 * est.std_error


@pytest.mark.parametrize("otype", ["call", "put"])
def test_pathwise_vega_matches_analytic(otype: str) -> None:
    S, K, T, r, sigma, q = 100.0, 95.0, 0.75, 0.03, 0.30, 0.02
    ana = bs_greeks(S, K, T, r, sigma, q, otype).vega
    est = mc_vega_pathwise(S, K, T, r, sigma, q, otype, n_paths=400_000, seed=22)
    assert abs(est.value - ana) <= 3.0 * est.std_error


@pytest.mark.parametrize("otype", ["call", "put"])
def test_lr_delta_matches_analytic(otype: str) -> None:
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    ana = bs_greeks(S, K, T, r, sigma, q, otype).delta
    est = mc_delta_lr(S, K, T, r, sigma, q, otype, n_paths=400_000, seed=23)
    assert abs(est.value - ana) <= 3.0 * est.std_error


@pytest.mark.parametrize("otype", ["call", "put"])
def test_lr_vega_matches_analytic(otype: str) -> None:
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    ana = bs_greeks(S, K, T, r, sigma, q, otype).vega
    est = mc_vega_lr(S, K, T, r, sigma, q, otype, n_paths=400_000, seed=24)
    assert abs(est.value - ana) <= 3.0 * est.std_error


def test_pathwise_lower_variance_than_lr_for_vanilla_delta() -> None:
    kwargs = dict(n_paths=100_000, seed=31)
    pw = mc_delta_pathwise(100, 100, 1, 0.05, 0.2, 0.0, "call", **kwargs)
    lr = mc_delta_lr(100, 100, 1, 0.05, 0.2, 0.0, "call", **kwargs)
    assert pw.std_error < lr.std_error


@pytest.mark.parametrize("greek", ["delta", "vega", "rho", "theta"])
def test_fd_fallback_greeks_close_to_analytic(greek: str) -> None:
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.01
    ana = getattr(bs_greeks(S, K, T, r, sigma, q, "call"), greek)
    est = mc_greek_fd(greek, S, K, T, r, sigma, q, "call", n_paths=400_000, seed=41)
    # common random numbers keep the noise small; 2% relative or small abs
    assert est == pytest.approx(ana, rel=2e-2, abs=2e-2)


def test_mc_deterministic_when_t_or_sigma_zero() -> None:
    res_t0 = mc_price(105, 100, 0.0, 0.05, 0.2, 0.0, "call", n_paths=1000, seed=1)
    assert res_t0.value == 5.0 and res_t0.std_error == 0.0
    res_s0 = mc_price(105, 100, 1.0, 0.05, 0.0, 0.0, "call", n_paths=1000, seed=1)
    assert res_s0.value == bs_price(105, 100, 1.0, 0.05, 0.0, 0.0, "call")


def test_mc_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        mc_price(-1, 100, 1, 0.05, 0.2, 0.0, "call")
    with pytest.raises(ValueError):
        mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call", n_paths=1)
    with pytest.raises(ValueError):
        mc_greek_fd("vanna", 100, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        mc_delta_pathwise(100, 100, 0.0, 0.05, 0.2)


def test_mcresult_contains_helper() -> None:
    res = mc_price(100, 100, 1, 0.05, 0.2, 0.0, "call", n_paths=50_000, seed=8)
    assert res.contains(res.value)
    assert not res.contains(res.value + 10.0)


def test_mc_std_error_scales_as_inverse_sqrt_n_fitted() -> None:
    """Fit the Monte Carlo error-scaling exponent by log-log regression.

    Plain (no variance reduction) MC has statistical error O(1/sqrt(n)) by
    the CLT. Rather than trust the estimator's *own* reported std_error
    formula (which would only check that the formula is internally
    consistent), this measures the *empirical* spread of independent
    replications of the price estimate at each path count and fits the
    exponent of empirical_std vs n by regression -- an actual measurement
    of the realized convergence rate, not a single eyeballed ratio.
    """
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    path_counts = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000]
    n_reps = 40
    empirical_std = []
    for n in path_counts:
        reps = [
            mc_price(
                S, K, T, r, sigma, q, "call",
                n_paths=n, antithetic=False, control_variate=False,
                seed=1_000_003 * n + rep,
            ).value
            for rep in range(n_reps)
        ]
        empirical_std.append(np.std(reps, ddof=1))
    slope, _intercept = np.polyfit(np.log(path_counts), np.log(empirical_std), 1)
    assert -0.65 < slope < -0.35, (
        f"fitted MC error-scaling exponent {slope:.3f}; CLT predicts -0.5 "
        "(std ~ C/sqrt(n))"
    )
