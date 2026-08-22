"""CRR binomial tree: convergence to GK, American exercise economics."""

import numpy as np
import pytest

from fx_options import binomial_convergence, binomial_price, gk_price

MKT = dict(S=1.10, K=1.12, T=0.5, r_d=0.0425, r_f=0.0290, sigma=0.0825)


class TestEuropeanConvergence:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_converges_to_gk(self, option_type):
        tree = binomial_price(**MKT, option_type=option_type, steps=2000)
        gk = gk_price(**MKT, option_type=option_type)
        assert tree == pytest.approx(gk, abs=5e-6)

    def test_error_decreases_with_steps(self):
        rows = binomial_convergence(**MKT, option_type="call",
                                    step_grid=(10, 100, 1000))
        errors = [r["abs_error"] for r in rows]
        assert errors[2] < errors[0]
        assert errors[2] < 1e-5

    def test_convergence_rate_fitted_exponent(self):
        """Fit the CRR discretisation-error exponent by log-log regression.

        CRR is a first-order scheme: error(n) ~ C / n, with an odd/even
        and node-alignment oscillation superposed. A single pair of step
        counts (as in ``test_error_decreases_with_steps`` above) cannot
        *prove* the O(1/n) rate -- a two-point ratio can land anywhere in
        the oscillation. Regressing log|error| on log(n) over a decade of
        geometrically spaced step counts averages the oscillation out and
        recovers the leading exponent, which must come out close to the
        theoretical -1.
        """
        steps = tuple(200 * 2**k for k in range(9))  # 200 .. 51200
        rows = binomial_convergence(**MKT, option_type="call", step_grid=steps)
        errs = np.array([r["abs_error"] for r in rows])
        assert np.all(errs > 0.0), "tree exactly matches GK at some n"
        slope, _ = np.polyfit(np.log(steps), np.log(errs), 1)
        assert -1.3 < slope < -0.7, (
            f"fitted CRR convergence exponent {slope:.3f}; theory predicts "
            "-1 (error ~ C/n)"
        )

    def test_jpy_level_convergence(self):
        # High spot level (pip 0.01) must not degrade accuracy in relative terms.
        args = dict(S=147.5, K=145.0, T=0.5, r_d=0.005, r_f=0.0525,
                    sigma=0.1075)
        tree = binomial_price(**args, option_type="put", steps=2000)
        gk = gk_price(**args, option_type="put")
        assert tree == pytest.approx(gk, rel=3e-4)


class TestAmerican:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_american_geq_european(self, option_type):
        eu = binomial_price(**MKT, option_type=option_type, steps=500,
                            exercise="european")
        am = binomial_price(**MKT, option_type=option_type, steps=500,
                            exercise="american")
        assert am >= eu - 1e-12

    def test_american_call_premium_positive_when_rf_above_rd(self):
        # Foreign carry (r_f > r_d) makes early exercise of a call on the
        # base currency optimal: USDJPY-style. Economically the option
        # holder forgoes the high USD deposit rate while unexercised.
        args = dict(S=147.5, K=140.0, T=1.0, r_d=0.005, r_f=0.0525,
                    sigma=0.1075)
        eu = binomial_price(**args, option_type="call", steps=1000,
                            exercise="european")
        am = binomial_price(**args, option_type="call", steps=1000,
                            exercise="american")
        assert am - eu > 1e-3  # economically significant premium

    def test_american_call_no_premium_when_rf_zero_ish(self):
        # With r_f <= r_d an American FX call is worth its European value
        # (no carry incentive), like a no-dividend equity call.
        eu = binomial_price(**MKT, option_type="call", steps=500,
                            exercise="european")
        am = binomial_price(**MKT, option_type="call", steps=500,
                            exercise="american")
        assert am == pytest.approx(eu, abs=1e-10)

    def test_american_put_premium_when_rd_high(self):
        args = dict(S=1.10, K=1.15, T=1.0, r_d=0.08, r_f=0.00, sigma=0.10)
        eu = binomial_price(**args, option_type="put", steps=1000,
                            exercise="european")
        am = binomial_price(**args, option_type="put", steps=1000,
                            exercise="american")
        assert am - eu > 1e-4

    def test_american_at_least_intrinsic(self):
        am = binomial_price(S=1.30, K=1.10, T=0.5, r_d=0.02, r_f=0.05,
                            sigma=0.10, option_type="call", steps=200,
                            exercise="american")
        assert am >= 0.20 - 1e-12


class TestLimitsAndValidation:
    def test_t_zero_intrinsic(self):
        assert binomial_price(1.2, 1.1, 0.0, 0.03, 0.01, 0.1, "call",
                              steps=10) == pytest.approx(0.1)

    def test_sigma_zero_matches_gk_european(self):
        val = binomial_price(1.1, 1.05, 0.5, 0.04, 0.01, 0.0, "call",
                             steps=100)
        assert val == pytest.approx(
            gk_price(1.1, 1.05, 0.5, 0.04, 0.01, 0.0, "call"), abs=1e-14)

    def test_invalid_steps_raise(self):
        with pytest.raises(ValueError, match="steps"):
            binomial_price(**MKT, option_type="call", steps=0)
        with pytest.raises(ValueError, match="steps"):
            binomial_price(**MKT, option_type="call", steps=2.5)  # type: ignore[arg-type]

    def test_invalid_exercise_raises(self):
        with pytest.raises(ValueError, match="exercise"):
            binomial_price(**MKT, option_type="call", steps=10,
                           exercise="bermudan")

    def test_invalid_market_inputs_raise(self):
        with pytest.raises(ValueError):
            binomial_price(-1.0, 1.1, 0.5, 0.03, 0.01, 0.1, "call")
