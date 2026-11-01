"""Cross-validation against the independent `arch` package (K. Sheppard).

All models here are implemented from scratch. `arch` is used only as an
external benchmark and is an optional test dependency (``pip install
.[test]``). Without it this module is skipped, not the whole suite.

Scaling: `arch` is fed percent returns (100x). Then omega_pct = 1e4 *
omega_decimal for GARCH/GJR (alpha, beta, gamma invariant), omega_pct =
omega_dec + (1 - beta) ln(1e4) for EGARCH (log-variance recursion), and the
log-likelihoods differ by exactly n ln(100). Our recursions use arch's
backcast and one-step-from-backcast initialisation (EGARCH included), so at
identical parameters the likelihoods agree to round-off.

Standard errors are compared like for like: our classic SEs against arch's
``cov_type="classic"``, our Bollerslev-Wooldridge robust SEs against arch's
``cov_type="robust"``, on Student-t data where the two differ. Both sides
are central finite differences of the same likelihood at optima that agree
to ~1e-6 in the parameters. Measured agreement is ~1e-4 relative, so a 1e-2
relative tolerance keeps two orders of magnitude of slack and is still 50x
tighter than the classic-vs-robust gap on this data.
"""

import numpy as np
import pytest

arch = pytest.importorskip("arch")
from arch import arch_model  # noqa: E402

from eq_vol.data import synthetic as syn  # noqa: E402
from eq_vol.egarch import egarch_recursion, expected_abs_z, fit_egarch  # noqa: E402
from eq_vol.forecasting import forecast_garch, forecast_gjr  # noqa: E402
from eq_vol.garch import fit_garch, garch_loglik, gaussian_loglik  # noqa: E402
from eq_vol.gjr import fit_gjr  # noqa: E402

N = 5000
LOG100 = np.log(100.0)


@pytest.fixture(scope="module")
def garch_pair():
    r = syn.simulate_garch(N, seed=42).returns
    ours = fit_garch(r)
    theirs = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    return r, ours, theirs


@pytest.fixture(scope="module")
def qmle_on_t_data():
    """Gaussian QMLE on Student-t(5) data: classic and robust SEs differ."""
    r = syn.simulate_garch(N, dist="t", nu=5.0, seed=7).returns
    ours = fit_garch(r)
    model = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    return ours, model.fit(disp="off", cov_type="classic"), model.fit(disp="off", cov_type="robust")


class TestGARCHCrossCheck:
    def test_parameters_agree(self, garch_pair):
        _, ours, theirs = garch_pair
        assert ours.params["omega"] * 1e4 == pytest.approx(theirs.params["omega"], rel=1e-3)
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=5e-4)
        assert ours.params["beta"] == pytest.approx(theirs.params["beta[1]"], abs=5e-4)

    def test_loglikelihood_agrees_up_to_jacobian(self, garch_pair):
        _, ours, theirs = garch_pair
        assert ours.loglik == pytest.approx(theirs.loglikelihood + N * LOG100, abs=1e-3)

    def test_likelihood_identity_at_arch_params(self, garch_pair):
        r, _, theirs = garch_pair
        ll = garch_loglik(
            r,
            omega=theirs.params["omega"] / 1e4,
            alpha=theirs.params["alpha[1]"],
            beta=theirs.params["beta[1]"],
            init_method="backcast",
        )
        assert ll == pytest.approx(theirs.loglikelihood + N * LOG100, abs=1e-6)

    def test_student_t_nu_agrees(self):
        r = syn.simulate_garch(N, dist="t", nu=8.0, seed=45).returns
        ours = fit_garch(r, dist="t")
        theirs = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        assert ours.params["nu"] == pytest.approx(theirs.params["nu"], rel=0.05)
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=2e-3)


class TestStandardErrorsLikeForLike:
    @pytest.mark.parametrize("ours_k,theirs_k", [("alpha", "alpha[1]"), ("beta", "beta[1]")])
    def test_classic_matches_arch_classic(self, qmle_on_t_data, ours_k, theirs_k):
        ours, classic, _ = qmle_on_t_data
        assert ours.std_errors[ours_k] == pytest.approx(classic.std_err[theirs_k], rel=1e-2)

    @pytest.mark.parametrize("ours_k,theirs_k", [("alpha", "alpha[1]"), ("beta", "beta[1]")])
    def test_robust_matches_arch_robust(self, qmle_on_t_data, ours_k, theirs_k):
        ours, _, robust = qmle_on_t_data
        assert ours.robust_std_errors[ours_k] == pytest.approx(robust.std_err[theirs_k], rel=1e-2)

    def test_omega_robust_matches_after_unit_map(self, qmle_on_t_data):
        ours, _, robust = qmle_on_t_data
        assert ours.robust_std_errors["omega"] * 1e4 == pytest.approx(
            robust.std_err["omega"], rel=1e-2
        )

    def test_robust_exceeds_classic_on_fat_tails(self, qmle_on_t_data):
        # Under t(5) innovations the information-matrix equality fails and
        # the sandwich SE is ~1.5x the classic SE (measured 1.57 alpha, 1.47 beta).
        ours, _, _ = qmle_on_t_data
        for k in ("alpha", "beta"):
            assert ours.robust_std_errors[k] > 1.3 * ours.std_errors[k]


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
    @pytest.fixture(scope="class")
    @classmethod
    def pair(cls):
        r = syn.simulate_egarch(N, seed=44).returns
        ours = fit_egarch(r)
        theirs = arch_model(100 * r, mean="Zero", vol="EGARCH", p=1, o=1, q=1).fit(disp="off")
        return r, ours, theirs

    def test_parameters_agree(self, pair):
        _, ours, theirs = pair
        omega_theirs_dec = theirs.params["omega"] - (1.0 - theirs.params["beta[1]"]) * np.log(1e4)
        assert ours.params["omega"] == pytest.approx(omega_theirs_dec, abs=5e-3)
        assert ours.params["alpha"] == pytest.approx(theirs.params["alpha[1]"], abs=5e-3)
        assert ours.params["gamma"] == pytest.approx(theirs.params["gamma[1]"], abs=5e-3)
        assert ours.params["beta"] == pytest.approx(theirs.params["beta[1]"], abs=5e-3)

    def test_likelihood_identity_at_arch_params(self, pair):
        # Same initialisation as arch (pre-sample shock at its expectation),
        # so our likelihood at arch's optimum equals arch's to round-off.
        r, ours, theirs = pair
        p = theirs.params
        omega = p["omega"] - (1.0 - p["beta[1]"]) * np.log(1e4)
        s2 = egarch_recursion(
            r, omega, p["alpha[1]"], p["gamma[1]"], p["beta[1]"], ours.init_var, expected_abs_z()
        )
        assert gaussian_loglik(r, s2) == pytest.approx(theirs.loglikelihood + N * LOG100, abs=1e-6)

    def test_optimum_at_least_as_good(self, pair):
        _, ours, theirs = pair
        assert ours.loglik >= theirs.loglikelihood + N * LOG100 - 1e-3


class TestForecastCrossCheck:
    """Analytic multi-step forecasts at identical parameters."""

    def test_garch_forecast_matches_arch(self, garch_pair):
        r, ours, _ = garch_pair
        p = ours.params
        fixed = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1).fix(
            [p["omega"] * 1e4, p["alpha"], p["beta"]]
        )
        theirs = fixed.forecast(horizon=10, reindex=False).variance.to_numpy()[-1] / 1e4
        np.testing.assert_allclose(forecast_garch(ours, 10), theirs, rtol=1e-10)

    def test_gjr_forecast_matches_arch(self):
        r = syn.simulate_gjr(N, seed=43).returns
        ours = fit_gjr(r)
        p = ours.params
        fixed = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, o=1, q=1).fix(
            [p["omega"] * 1e4, p["alpha"], p["gamma"], p["beta"]]
        )
        theirs = fixed.forecast(horizon=10, reindex=False).variance.to_numpy()[-1] / 1e4
        np.testing.assert_allclose(forecast_gjr(ours, 10), theirs, rtol=1e-10)
