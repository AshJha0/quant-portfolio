"""RiskMetrics EWMA variance filter."""

import numpy as np
import pytest

from eq_vol.data import synthetic as syn
from eq_vol.ewma import (
    ewma_forecast,
    ewma_variance,
    ewma_variance_recursive,
    ewma_weights,
    halflife_to_lambda,
    lambda_to_halflife,
)


@pytest.fixture(scope="module")
def returns():
    return syn.simulate_garch(1000, seed=40).returns


class TestRecursion:
    def test_vectorised_matches_recursive_exactly(self, returns):
        for lam in (0.90, 0.94, 0.97):
            np.testing.assert_allclose(
                ewma_variance(returns, lam=lam),
                ewma_variance_recursive(returns, lam=lam),
                rtol=1e-13,
                atol=1e-20,
            )

    def test_matches_brute_force_weighted_sum(self, returns):
        lam, v0 = 0.94, 1e-4
        sigma2 = ewma_variance(returns, lam=lam, init_var=v0)
        for t in (1, 5, 50, 500, 999):
            w = ewma_weights(t, lam)
            brute = float(w @ (returns[t - 1 :: -1] ** 2)) + lam**t * v0
            assert sigma2[t] == pytest.approx(brute, rel=1e-10)

    def test_weights_sum_to_one_with_tail(self):
        n, lam = 100, 0.94
        assert ewma_weights(n, lam).sum() + lam**n == pytest.approx(1.0, abs=1e-12)

    def test_lambda_one_is_constant_initial_variance(self, returns):
        v0 = 2.5e-4
        sigma2 = ewma_variance(returns, lam=1.0, init_var=v0)
        np.testing.assert_allclose(sigma2, v0)

    def test_monotone_response_to_shock_size(self):
        base = np.full(50, 0.005)
        for shock_small, shock_big in [(0.02, 0.05), (0.05, 0.10)]:
            r_small = np.append(base, shock_small)
            r_big = np.append(base, shock_big)
            v_small = ewma_forecast(r_small, 1)[0]
            v_big = ewma_forecast(r_big, 1)[0]
            assert v_big > v_small

    def test_shock_then_geometric_decay(self):
        # after a single large shock followed by zero returns, the variance
        # decays by exactly lambda per day
        lam = 0.94
        r = np.concatenate([np.full(30, 0.01), [0.10], np.zeros(20)])
        sigma2 = ewma_variance(r, lam=lam)
        post = sigma2[32:]  # from the first post-shock day onward
        np.testing.assert_allclose(post[1:] / post[:-1], lam, rtol=1e-12)

    def test_shock_raises_next_day_variance(self):
        r = np.concatenate([np.full(30, 0.005), [0.08]])
        sigma2 = ewma_variance(np.append(r, 0.0))
        assert sigma2[-1] > sigma2[-2]


class TestHalflife:
    def test_riskmetrics_halflife(self):
        assert lambda_to_halflife(0.94) == pytest.approx(11.2, abs=0.1)

    def test_roundtrip(self):
        for h in (5.0, 11.2, 30.0):
            assert lambda_to_halflife(halflife_to_lambda(h)) == pytest.approx(h, rel=1e-12)

    def test_halflife_meaning(self):
        lam = halflife_to_lambda(10.0)
        assert lam**10 == pytest.approx(0.5, rel=1e-12)

    def test_lambda_one_infinite_halflife(self):
        assert lambda_to_halflife(1.0) == np.inf

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            lambda_to_halflife(0.0)
        with pytest.raises(ValueError):
            halflife_to_lambda(-1.0)
        with pytest.raises(ValueError):
            ewma_variance(np.array([0.01, 0.02]), lam=1.5)


class TestForecast:
    def test_flat_term_structure(self, returns):
        f = ewma_forecast(returns, horizon=50)
        assert f.shape == (50,)
        np.testing.assert_allclose(f, f[0])

    def test_one_step_equals_recursion(self, returns):
        lam = 0.94
        sigma2 = ewma_variance(returns, lam=lam)
        expected = lam * sigma2[-1] + (1 - lam) * returns[-1] ** 2
        assert ewma_forecast(returns, 1, lam=lam)[0] == pytest.approx(expected, rel=1e-14)

    def test_invalid_horizon_raises(self, returns):
        with pytest.raises(ValueError, match="horizon"):
            ewma_forecast(returns, horizon=0)
