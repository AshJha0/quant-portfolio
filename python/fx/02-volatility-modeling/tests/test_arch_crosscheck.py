"""Cross-validation of the from-scratch MLE against the `arch` package.

`arch` is used here only; the library never imports it. It is an optional
test dependency (``pip install .[test]``): without it this module is
skipped and the rest of the suite still runs.

arch is fed 100 * r (its recommended scaling), so for GARCH/GJR its omega is
1e4 times ours, for EGARCH omega_pct = omega_dec + (1 - beta) ln(1e4), and
the log-likelihood differs by the change-of-variables constant n log(100).

Standard errors are compared like for like (classic vs ``cov_type="classic"``,
robust vs ``cov_type="robust"``) on Student-t data, where the two differ.
Both are central finite differences of the same likelihood at optima that
agree to ~1e-5 in the parameters; measured agreement is ~1e-3 relative or
better, so 2e-2 relative keeps an order of magnitude of slack and is still
far tighter than the ~40% classic-vs-robust gap on this data.
"""

import numpy as np
import pytest

pytest.importorskip("arch")
from arch import arch_model  # noqa: E402

from fx_vol import fit_egarch, fit_garch, fit_gjr, forecast_variance  # noqa: E402
from fx_vol._mle import gaussian_loglik  # noqa: E402
from fx_vol.data import synthetic as syn  # noqa: E402
from fx_vol.egarch import egarch_filter  # noqa: E402

SCALE = 100.0


@pytest.fixture(scope="module")
def crosscheck_series():
    return syn.simulate_garch(5000, omega=1e-6, alpha=0.05, beta=0.92, seed=71)


@pytest.fixture(scope="module")
def crosscheck_series_t():
    return syn.simulate_garch(5000, omega=1.5e-6, alpha=0.06, beta=0.90, dist="t", nu=6.0, seed=72)


@pytest.fixture(scope="module")
def qmle_on_t(crosscheck_series_t):
    r = crosscheck_series_t
    ours = fit_garch(r, dist="gaussian")
    m = arch_model(SCALE * r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    return ours, m.fit(disp="off", cov_type="classic"), m.fit(disp="off", cov_type="robust")


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


class TestStandardErrorsLikeForLike:
    @pytest.mark.parametrize("k,ak", [("alpha", "alpha[1]"), ("beta", "beta[1]")])
    def test_classic(self, qmle_on_t, k, ak):
        ours, classic, _ = qmle_on_t
        assert ours.std_errors[k] == pytest.approx(classic.std_err[ak], rel=2e-2)

    @pytest.mark.parametrize("k,ak", [("alpha", "alpha[1]"), ("beta", "beta[1]")])
    def test_robust(self, qmle_on_t, k, ak):
        ours, _, robust = qmle_on_t
        assert ours.robust_std_errors[k] == pytest.approx(robust.std_err[ak], rel=2e-2)

    def test_omega_robust_after_unit_map(self, qmle_on_t):
        ours, _, robust = qmle_on_t
        assert ours.robust_std_errors["omega"] * SCALE ** 2 == pytest.approx(
            robust.std_err["omega"], rel=2e-2
        )

    def test_robust_exceeds_classic(self, qmle_on_t):
        ours, _, _ = qmle_on_t
        assert ours.robust_std_errors["alpha"] > 1.2 * ours.std_errors["alpha"]


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
        assert ours.loglik >= mapped - 0.05


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

    def test_gjr_forecast_matches_arch(self):
        r = syn.simulate_gjr(5000, omega=1e-6, alpha=0.03, gamma=0.10, beta=0.88, seed=73)
        ours = fit_gjr(r, dist="gaussian")
        p = ours.params
        fixed = arch_model(SCALE * r, mean="Zero", vol="GARCH", p=1, o=1, q=1).fix(
            [p["omega"] * SCALE ** 2, p["alpha"], p["gamma"], p["beta"]]
        )
        theirs = fixed.forecast(horizon=10, reindex=False).variance.to_numpy()[-1] / SCALE ** 2
        np.testing.assert_allclose(forecast_variance(ours, 10), theirs, rtol=1e-10)


class TestEgarchCrosscheck:
    @pytest.fixture(scope="class")
    @classmethod
    def pair(cls):
        r = syn.simulate_egarch(5000, omega=-0.5, alpha=0.15, gamma=-0.06, beta=0.95, seed=74)
        ours = fit_egarch(r, dist="gaussian")
        m = arch_model(SCALE * r, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal")
        return r, ours, m.fit(disp="off", cov_type="classic")

    def test_params_match(self, pair):
        _, ours, theirs = pair
        omega_dec = theirs.params["omega"] - (1.0 - theirs.params["beta[1]"]) * np.log(SCALE ** 2)
        assert ours.params["omega"] == pytest.approx(omega_dec, abs=5e-3)
        for k, ak in (("alpha", "alpha[1]"), ("gamma", "gamma[1]"), ("beta", "beta[1]")):
            assert ours.params[k] == pytest.approx(theirs.params[ak], abs=5e-3)

    def test_loglik_identity_at_arch_params(self, pair):
        # Same initialisation as arch (pre-sample shock at its expectation),
        # so the likelihoods agree to round-off at identical parameters.
        r, ours, theirs = pair
        p = theirs.params
        omega = p["omega"] - (1.0 - p["beta[1]"]) * np.log(SCALE ** 2)
        s2 = egarch_filter(r, omega, p["alpha[1]"], p["gamma[1]"], p["beta[1]"],
                           initial_variance=ours.extra["backcast"])
        assert gaussian_loglik(r, s2) == pytest.approx(
            theirs.loglikelihood + r.size * np.log(SCALE), abs=1e-6
        )

    def test_omega_se_matches_arch_in_percent_units(self, pair):
        # Fitting 100 r makes our omega the same parameter as arch's. The old
        # code reported the scaled-data SE of omega unchanged (no delta
        # method for the beta-dependent shift): ~3x too small in percent
        # units here and ~34x too small on decimal returns.
        r, _, theirs = pair
        ours_pct = fit_egarch(SCALE * r, dist="gaussian")
        assert ours_pct.std_errors["omega"] == pytest.approx(theirs.std_err["omega"], rel=2e-2)
        assert ours_pct.std_errors["beta"] == pytest.approx(theirs.std_err["beta[1]"], rel=2e-2)
