"""Cross-validation against the independent `arch` package (K. Sheppard).

All models here are implemented from scratch; `arch` is used purely as an
external benchmark. Scaling convention: `arch` recommends percent returns
(100x), under which omega_pct = 1e4 * omega_decimal (alpha/beta invariant for
GARCH/GJR) and log-likelihoods differ by exactly n*ln(100) (a Jacobian term).
For EGARCH, omega_pct = omega_dec + (1 - beta) * ln(1e4) because the recursion
is in log-variance. Our recursions use arch's backcast initialisation, so on
identical parameters the likelihoods agree to machine precision.
"""

import numpy as np
import pytest
from arch import arch_model

from eq_vol.data import synthetic as syn
from eq_vol.egarch import fit_egarch
from eq_vol.garch import fit_garch, garch_loglik
from eq_vol.gjr import fit_gjr

N = 5000
LOG100 = np.log(100.0)


@pytest.fixture(scope="module")
def garch_pair():
    r = syn.simulate_garch(N, seed=42).returns
    ours = fit_garch(r)
    theirs = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    return r, ours, theirs


class TestGARCHCrossCheck:
    def test_parameters_agree(self, garch_pair):
        _, ours, theirs = garch_pair
        assert ours.params["omega"] * 1e4 == pytest.approx(theirs.params["omega"], rel=1e-3)
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=5e-4)
        assert ours.params["beta"] == pytest.approx(theirs.params["beta[1]"], abs=5e-4)

    def test_loglikelihood_agrees_up_to_jacobian(self, garch_pair):
        r, ours, theirs = garch_pair
        assert ours.loglik == pytest.approx(theirs.loglikelihood + N * LOG100, abs=1e-3)

    def test_likelihood_identity_at_arch_params(self, garch_pair):
        # evaluate OUR likelihood at ARCH's fitted parameters: agreement to
        # ~machine precision proves the recursion + likelihood are identical
        r, _, theirs = garch_pair
        ll = garch_loglik(
            r,
            omega=theirs.params["omega"] / 1e4,
            alpha=theirs.params["alpha[1]"],
            beta=theirs.params["beta[1]"],
            init_method="backcast",
        )
        assert ll == pytest.approx(theirs.loglikelihood + N * LOG100, abs=1e-6)

    def test_standard_errors_same_order(self, garch_pair):
        _, ours, theirs = garch_pair
        assert ours.std_errors["alpha"] == pytest.approx(theirs.std_err["alpha[1]"], rel=0.15)
        assert ours.std_errors["beta"] == pytest.approx(theirs.std_err["beta[1]"], rel=0.15)

    def test_student_t_nu_agrees(self):
        r = syn.simulate_garch(N, dist="t", nu=8.0, seed=45).returns
        ours = fit_garch(r, dist="t")
        theirs = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        assert ours.params["nu"] == pytest.approx(theirs.params["nu"], rel=0.05)
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=2e-3)


class TestGJRCrossCheck:
    def test_parameters_and_loglik_agree(self):
        r = syn.simulate_gjr(N, seed=43).returns
        ours = fit_gjr(r)
        theirs = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, o=1, q=1).fit(disp="off")
        assert ours.params["omega"] * 1e4 == pytest.approx(theirs.params["omega"], rel=1e-3)
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=1e-3)
        assert ours.params["gamma"] == pytest.approx(theirs.params["gamma[1]"], abs=1e-3)
        assert ours.params["beta"] == pytest.approx(theirs.params["beta[1]"], abs=1e-3)
        assert ours.loglik == pytest.approx(theirs.loglikelihood + N * LOG100, abs=1e-2)


class TestEGARCHCrossCheck:
    def test_parameters_and_loglik_agree(self):
        r = syn.simulate_egarch(N, seed=44).returns
        ours = fit_egarch(r)
        theirs = arch_model(100 * r, mean="Zero", vol="EGARCH", p=1, o=1, q=1).fit(disp="off")
        # log-variance scale shift: omega_pct = omega_dec + (1 - beta) ln(1e4)
        omega_theirs_dec = theirs.params["omega"] - (1.0 - theirs.params["beta[1]"]) * np.log(1e4)
        assert ours.params["omega"] == pytest.approx(omega_theirs_dec, abs=5e-3)
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=5e-3)
        assert ours.params["gamma"] == pytest.approx(theirs.params["gamma[1]"], abs=5e-3)
        assert ours.params["beta"] == pytest.approx(theirs.params["beta[1]"], abs=5e-3)
        # initialisation conventions differ slightly for EGARCH -> looser tol
        assert ours.loglik == pytest.approx(theirs.loglikelihood + N * LOG100, abs=0.5)
