"""Tests for the shared research statistics (identical across projects).

Monte Carlo tolerances are set from the Monte Carlo standard error of the
quantity being checked, stated next to each assertion.
"""

import math

import numpy as np
import pytest
from eq_regime.research_stats import (
    TrialLog,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    lo_sharpe_se,
    newey_west_lag,
    pbo_cscv,
    seed_distribution,
    sharpe_ratio,
)
from hypothesis import given, settings
from hypothesis import strategies as hst
from scipy.signal import lfilter


def test_newey_west_lag_rule():
    assert newey_west_lag(100) == 4
    assert newey_west_lag(1000) == 6


def test_lo_se_iid_matches_textbook_formula():
    # lag 0 on any series: (1 + s^2/2 * (kurtosis-1)/2 ...) -> for the exact
    # influence function the Gaussian value is (1 + s^2/2)/n; check on a
    # large Gaussian sample where sample moments are close to Gaussian.
    rng = np.random.default_rng(1)
    r = 0.05 + rng.standard_normal(200_000)
    s = r.mean() / r.std()
    se = lo_sharpe_se(r, periods_per_year=1, lag=0)
    assert se == pytest.approx(math.sqrt((1 + s * s / 2) / r.size), rel=0.01)


def test_lo_se_matches_monte_carlo_under_autocorrelation():
    # AR(1) phi=0.4 returns: the iid SE is too small. The NW SE must match the
    # Monte Carlo sd of the Sharpe estimate. 1000 replications give the MC sd a
    # relative SE of about 1/sqrt(2*1000) = 2.2%; NW bias at n=2000 is a few %.
    rng = np.random.default_rng(7)
    n, reps, phi = 2000, 1000, 0.4
    srs, ses = [], []
    for _ in range(reps):
        e = rng.standard_normal(n)
        x = lfilter([1.0], [1.0, -phi], e)
        r = 0.05 + x
        srs.append(r.mean() / r.std())
        ses.append(lo_sharpe_se(r, periods_per_year=1, lag=20))
    mc_sd = float(np.std(srs, ddof=1))
    assert np.mean(ses) == pytest.approx(mc_sd, rel=0.10)
    iid = lo_sharpe_se(r, periods_per_year=1, lag=0)
    assert iid < 0.8 * mc_sd


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(3)
    n_trials, v = 50, 0.01
    sims = rng.normal(0, math.sqrt(v), size=(20_000, n_trials)).max(axis=1)
    # the formula is an asymptotic approximation; within 3% at N=50
    assert expected_max_sharpe(n_trials, v) == pytest.approx(sims.mean(), rel=0.03)
    assert expected_max_sharpe(1, v) == 0.0


def test_deflated_sharpe_falls_with_trial_count():
    rng = np.random.default_rng(4)
    r = 0.10 + rng.standard_normal(1000)  # per-period SR about 0.1, t about 3
    d1 = deflated_sharpe_ratio(r, 1)
    d100 = deflated_sharpe_ratio(r, 100)
    assert d1 > 0.99
    assert d100 < d1


def test_deflated_sharpe_is_calibrated_under_the_null():
    # Under H0 (true SR 0, one trial) the DSR is a p-value-like quantity that is
    # uniform; P(DSR > 0.95) should be 5%. 1500 draws: SE 0.56%, bound 3.5 SE.
    rng = np.random.default_rng(5)
    hits = sum(deflated_sharpe_ratio(rng.standard_normal(500), 1) > 0.95 for _ in range(1500))
    assert abs(hits / 1500 - 0.05) < 0.02


def test_pbo_noise_is_near_half_and_skill_near_zero():
    rng = np.random.default_rng(6)
    # PBO of one noise matrix is itself very noisy (0.17 to 0.84 over 12
    # draws); the mean over 30 matrices is near 0.5 (sd per draw about 0.2,
    # so SE of the mean about 0.04).
    vals = [pbo_cscv(rng.standard_normal((800, 10)), n_splits=8) for _ in range(30)]
    assert 0.38 < np.mean(vals) < 0.62
    noise = rng.standard_normal((1600, 10))
    skilled = noise.copy()
    skilled[:, 3] += 0.3
    assert pbo_cscv(skilled, n_splits=10) < 0.05


def test_block_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(8)
    covered = 0
    for k in range(200):
        r = 0.1 + rng.standard_normal(300)
        _, lo, hi = block_bootstrap_ci(r, np.mean, n_boot=300, seed=k)
        covered += lo <= 0.1 <= hi
    # nominal 95%; percentile bootstrap on n=300 under-covers slightly.
    # 200 trials: binomial SE about 1.5%.
    assert covered / 200 > 0.89


def test_seed_distribution_matches_scipy():
    v = np.array([1.0, 2.0, 4.0, 3.0])
    s = seed_distribution(v)
    assert s.mean == 2.5 and s.n == 4
    from scipy import stats

    lo, hi = stats.t.interval(0.95, 3, loc=2.5, scale=stats.sem(v))
    assert s.ci_low == pytest.approx(lo) and s.ci_high == pytest.approx(hi)


def test_trial_log_counts():
    log = TrialLog()
    log.log("a", 0.1, window=5)
    log.log("b", None)
    assert log.n_trials == 2 and log.sharpes() == [0.1]
    with pytest.raises(ValueError):
        log.log("c", float("nan"))


@pytest.mark.parametrize(
    "fn",
    [
        lambda x: sharpe_ratio(x),
        lambda x: lo_sharpe_se(x),
        lambda x: deflated_sharpe_ratio(x, 3),
        lambda x: block_bootstrap_ci(x, np.mean, n_boot=100),
        lambda x: seed_distribution(x),
    ],
)
def test_rejects_non_finite(fn):
    with pytest.raises(ValueError):
        fn(np.array([0.1, np.nan, 0.2, 0.3, 0.1]))
    with pytest.raises(ValueError):
        fn(np.array([0.1, np.inf, 0.2, 0.3, 0.1]))


def test_rejects_constant_series():
    with pytest.raises(ValueError):
        sharpe_ratio(np.ones(10))
    with pytest.raises(ValueError):
        lo_sharpe_se(np.ones(10))


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((100, 3)), n_splits=5)


@settings(max_examples=50, deadline=None)
@given(
    scale=hst.floats(0.01, 100.0),
    seed=hst.integers(0, 10_000),
)
def test_sharpe_and_se_are_scale_invariant(scale, seed):
    r = 0.02 + np.random.default_rng(seed).standard_normal(200)
    assert sharpe_ratio(scale * r) == pytest.approx(sharpe_ratio(r), rel=1e-9)
    assert lo_sharpe_se(scale * r) == pytest.approx(lo_sharpe_se(r), rel=1e-9)
