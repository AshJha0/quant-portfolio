"""EVT/POT: exact GPD quantile and tail mean, xi = 0 limit, MLE convergence."""

import numpy as np
import pytest
from eq_var.evt import fit_pot, mean_excess, pot_es, pot_var
from scipy.integrate import quad
from scipy.stats import genpareto


@pytest.mark.parametrize("xi", [-0.2, 0.0, 0.25, 0.6])
def test_pot_matches_exact_gpd_tail(xi):
    # Whole sample above u=0 (n_u = n): POT VaR is the GPD quantile and ES is
    # the mean of the quantile function over the tail.
    beta, a = 0.7, 0.01
    v = pot_var(a, 0.0, xi, beta, 1000, 1000)
    assert v == pytest.approx(genpareto.ppf(1 - a, xi, scale=beta), rel=1e-10)
    es_ref = quad(lambda p: genpareto.ppf(1 - p, xi, scale=beta), 0, a, epsabs=1e-13)[0] / a
    assert pot_es(a, 0.0, xi, beta, 1000, 1000) == pytest.approx(es_ref, rel=1e-8)


def test_threshold_shift():
    # Moving u by c with the same excess distribution moves VaR by c.
    assert pot_var(0.001, 2.0, 0.3, 1.0, 1000, 50) == pytest.approx(
        2.0 + pot_var(0.001, 0.0, 0.3, 1.0, 1000, 50))


def test_fit_recovers_parameters():
    rng = np.random.default_rng(13)
    losses = genpareto.rvs(0.3, scale=1.2, size=200_000, random_state=rng)
    tail = fit_pot(-losses, threshold=0.0)
    assert tail.xi == pytest.approx(0.3, abs=0.01)
    assert tail.beta == pytest.approx(1.2, rel=0.01)
    assert tail.var(0.001) == pytest.approx(genpareto.ppf(0.999, 0.3, scale=1.2), rel=0.03)


def test_student_t_tail_index():
    # t_4 has tail index 4, so the GPD shape above a high threshold is ~1/4.
    rng = np.random.default_rng(17)
    pnl = rng.standard_t(4, size=400_000)
    tail = fit_pot(pnl, 0.99)
    assert tail.xi == pytest.approx(0.25, abs=0.07)


def test_mean_excess_exponential_is_flat():
    rng = np.random.default_rng(19)
    pnl = -rng.exponential(2.0, size=200_000)
    np.testing.assert_allclose(mean_excess(pnl, [0.5, 1.0, 2.0]), 2.0, rtol=0.03)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        pot_var(0.2, 0.0, 0.1, 1.0, 100, 10)  # alpha outside fitted tail
    with pytest.raises(ValueError):
        pot_es(0.01, 0.0, 1.2, 1.0, 100, 100)
    with pytest.raises(ValueError):
        fit_pot(np.zeros(100), 0.99)
