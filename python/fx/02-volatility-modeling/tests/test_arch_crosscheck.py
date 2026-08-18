"""Cross-validation of the from-scratch MLE against the `arch` package.

`arch` is used HERE ONLY -- the library itself never imports it. The
comparison handles arch's percent-scaling convention explicitly: arch is fed
100 * r (its recommended scaling), so its omega is 1e4 times ours and its
log-likelihood differs by the change-of-variables constant n * log(100).
"""

import numpy as np
import pytest
from arch import arch_model

from fx_vol import fit_garch, fit_gjr
from fx_vol.data import synthetic as syn

SCALE = 100.0


@pytest.fixture(scope="module")
def crosscheck_series():
    return syn.simulate_garch(5000, omega=1e-6, alpha=0.05, beta=0.92, seed=71)


@pytest.fixture(scope="module")
def crosscheck_series_t():
    return syn.simulate_garch(5000, omega=1.5e-6, alpha=0.06, beta=0.90, dist="t", nu=6.0, seed=72)


class TestGaussianGarch:
    def test_params_match_arch(self, crosscheck_series):
        r = crosscheck_series
        ours = fit_garch(r, dist="gaussian")
        theirs = arch_model(SCALE * r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=1e-4)
        assert ours.params["beta"] == pytest.approx(theirs.params["beta[1]"], abs=1e-4)
        assert ours.params["omega"] == pytest.approx(theirs.params["omega"] / SCALE ** 2, rel=1e-3)

    def test_loglik_matches_arch_with_scaling_map(self, crosscheck_series):
        r = crosscheck_series
        ours = fit_garch(r, dist="gaussian")
        theirs = arch_model(SCALE * r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        mapped = theirs.loglikelihood + r.size * np.log(SCALE)
        assert ours.loglik == pytest.approx(mapped, abs=1e-2)

    def test_conditional_variance_path_matches(self, crosscheck_series):
        r = crosscheck_series
        ours = fit_garch(r, dist="gaussian")
        theirs = arch_model(SCALE * r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
        theirs_var = np.asarray(theirs.conditional_volatility) ** 2 / SCALE ** 2
        assert np.allclose(ours.sigma2, theirs_var, rtol=5e-3)


class TestStudentTGarch:
    def test_params_and_dof_match_arch(self, crosscheck_series_t):
        r = crosscheck_series_t
        ours = fit_garch(r, dist="t")
        theirs = arch_model(SCALE * r, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=5e-4)
        assert ours.params["beta"] == pytest.approx(theirs.params["beta[1]"], abs=5e-4)
        assert ours.params["omega"] == pytest.approx(theirs.params["omega"] / SCALE ** 2, rel=5e-3)
        assert ours.params["nu"] == pytest.approx(theirs.params["nu"], rel=0.02)
        mapped = theirs.loglikelihood + r.size * np.log(SCALE)
        assert ours.loglik >= mapped - 0.05  # our optimum at least as good, up to tolerance


class TestGjrCrosscheck:
    def test_gjr_params_match_arch(self):
        r = syn.simulate_gjr(5000, omega=1e-6, alpha=0.03, gamma=0.10, beta=0.88, seed=73)
        ours = fit_gjr(r, dist="gaussian")
        theirs = arch_model(SCALE * r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal").fit(disp="off")
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=5e-4)
        assert ours.params["gamma"] == pytest.approx(theirs.params["gamma[1]"], abs=1e-3)
        assert ours.params["beta"] == pytest.approx(theirs.params["beta[1]"], abs=5e-4)
        mapped = theirs.loglikelihood + r.size * np.log(SCALE)
        assert ours.loglik == pytest.approx(mapped, abs=0.05)
