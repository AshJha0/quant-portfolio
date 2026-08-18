"""Monte Carlo VaR: convergence, fat tails, jumps, Cholesky robustness."""

import numpy as np
import pandas as pd
import pytest

from fx_var import (
    Book,
    JumpSpec,
    NumericalWarning,
    Spot,
    monte_carlo_var,
    normal_var,
    parametric_var,
    robust_cholesky,
    simulate_factor_returns,
    var_standard_error,
)
from fx_var.data.synthetic import demo_em_book, demo_market, simulate_fx_returns

FACTORS = ["FX:EUR", "FX:GBP"]


@pytest.fixture()
def market():
    return demo_market()


@pytest.fixture()
def cov2():
    sd = np.array([0.006, 0.0055])
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    c = corr * np.outer(sd, sd)
    return pd.DataFrame(c, index=FACTORS, columns=FACTORS)


# ------------------------------------------------------------ convergence
def test_mc_matches_closed_form_linear_pnl(cov2):
    """Purely linear P&L w'x on simulated normal factors: MC VaR within
    3 SE of the exact normal closed form."""
    w = np.array([1.2e7, -0.8e7])
    scen = simulate_factor_returns(cov2, 200_000, dist="normal", seed=123)
    pnl = scen.to_numpy() @ w
    sigma = float(np.sqrt(w @ cov2.to_numpy() @ w))
    exact = normal_var(sigma, 0.99)
    from fx_var import empirical_var

    mc = empirical_var(pnl, 0.99)
    se = var_standard_error(pnl, 0.99)
    assert abs(mc - exact) < 3 * se


def test_mc_full_reval_converges_to_parametric(market, cov2):
    """Linear (spot-only) normal book: full-reval MC within 3 SE of the
    var-covar closed form (small exp() convexity allowance included)."""
    book = Book([Spot("EURUSD", 12e6), Spot("GBPUSD", -6e6)])
    rets = pd.DataFrame(
        simulate_factor_returns(cov2, 3000, dist="normal", seed=5),
        columns=FACTORS,
    )
    param = parametric_var(book, market, rets, 0.99)
    mc = monte_carlo_var(book, market, rets.cov(), 0.99, n_scenarios=200_000, seed=17)
    tol = 3 * mc.se_var + 0.01 * param.var  # 3 SE + 1% convexity allowance
    assert abs(mc.var - param.var) < tol
    assert mc.es >= mc.var


def test_mc_seed_reproducibility(market, cov2):
    book = Book([Spot("EURUSD", 1e7)])
    a = monte_carlo_var(book, market, cov2, 0.99, n_scenarios=5000, seed=42)
    b = monte_carlo_var(book, market, cov2, 0.99, n_scenarios=5000, seed=42)
    c = monte_carlo_var(book, market, cov2, 0.99, n_scenarios=5000, seed=43)
    assert a.var == b.var
    assert a.var != c.var


def test_var_se_shrinks_with_n(cov2):
    rng = np.random.default_rng(0)
    small = rng.standard_normal(2_000)
    large = rng.standard_normal(200_000)
    assert var_standard_error(large, 0.99) < var_standard_error(small, 0.99)
    assert var_standard_error(small, 0.99) > 0


# ------------------------------------------------------------ fat tails / EM
def _em_cov():
    rets = simulate_fx_returns(["MXN", "BRL", "TRY", "ZAR"], 1500, seed=21)
    return rets.cov()


def test_t_mc_exceeds_normal_mc_at_99_for_em_book(market):
    """Variance-matched Student-t factors: 99% VaR strictly above normal MC
    - normal MC underestimates EM tail risk at equal covariance."""
    book = demo_em_book()
    cov = _em_cov()
    n = monte_carlo_var(book, market, cov, 0.99, n_scenarios=100_000, dist="normal", seed=7)
    t5 = monte_carlo_var(book, market, cov, 0.99, n_scenarios=100_000, dist="t", df=5, seed=7)
    assert t5.var > n.var * 1.05
    assert t5.es > n.es * 1.10


def test_jump_mc_exceeds_normal_mc_at_99_for_em_book(market):
    book = demo_em_book()
    cov = _em_cov()
    jumps = JumpSpec(prob=0.02, mean={"FX:TRY": -0.15, "FX:BRL": -0.08},
                     std={"FX:TRY": 0.05, "FX:BRL": 0.03})
    n = monte_carlo_var(book, market, cov, 0.99, n_scenarios=100_000, dist="normal", seed=7)
    j = monte_carlo_var(book, market, cov, 0.99, n_scenarios=100_000, dist="jump",
                        jumps=jumps, seed=7)
    assert j.var > n.var * 1.10
    assert j.es > n.es


def test_t_variance_matched_to_cov(cov2):
    """The t simulator is scaled so sample covariance ~ target covariance."""
    scen = simulate_factor_returns(cov2, 300_000, dist="t", df=5, seed=3)
    got = scen.cov().to_numpy()
    np.testing.assert_allclose(got, cov2.to_numpy(), rtol=0.05)


def test_normal_sim_recovers_cov(cov2):
    scen = simulate_factor_returns(cov2, 200_000, dist="normal", seed=4)
    np.testing.assert_allclose(scen.cov().to_numpy(), cov2.to_numpy(), rtol=0.03)


# ------------------------------------------------------------ Cholesky
def test_robust_cholesky_pd_no_warning():
    import warnings

    a = np.array([[4.0, 1.0], [1.0, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        chol = robust_cholesky(a)
    np.testing.assert_allclose(chol @ chol.T, a, atol=1e-12)


def test_robust_cholesky_singular_pegs_warns_and_works():
    """Two perfectly correlated pegged currencies: singular covariance must
    factorise via jitter with a NumericalWarning, and the simulation must
    reproduce the (degenerate) covariance."""
    v = 1e-8  # peg-level daily variance
    cov = pd.DataFrame([[v, v], [v, v]], index=["FX:HKD", "FX:SAR"],
                       columns=["FX:HKD", "FX:SAR"])
    with pytest.warns(NumericalWarning, match="jitter"):
        chol = robust_cholesky(cov.to_numpy())
    np.testing.assert_allclose(chol @ chol.T, cov.to_numpy(), atol=1e-10)
    with pytest.warns(NumericalWarning):
        scen = simulate_factor_returns(cov, 50_000, seed=0)
    got = scen.cov().to_numpy()
    np.testing.assert_allclose(got, cov.to_numpy(), atol=2e-10)
    # the two pegs move in lockstep
    assert np.corrcoef(scen.iloc[:, 0], scen.iloc[:, 1])[0, 1] > 0.999


def test_robust_cholesky_invalid_inputs():
    with pytest.raises(ValueError, match="square"):
        robust_cholesky(np.ones((2, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        robust_cholesky(np.array([[1.0, 0.5], [0.1, 1.0]]))


# ------------------------------------------------------------ interface edges
def test_jump_requires_spec(cov2):
    with pytest.raises(ValueError, match="JumpSpec"):
        simulate_factor_returns(cov2, 100, dist="jump")


def test_invalid_dist_and_df(cov2):
    with pytest.raises(ValueError, match="dist"):
        simulate_factor_returns(cov2, 100, dist="levy")
    with pytest.raises(ValueError, match="df"):
        simulate_factor_returns(cov2, 100, dist="t", df=2.0)


def test_jump_spec_validation():
    with pytest.raises(ValueError):
        JumpSpec(prob=1.5, mean={"FX:TRY": -0.1})
    with pytest.raises(ValueError):
        JumpSpec(prob=0.1, mean={"FX:TRY": -0.1}, std={"FX:TRY": -0.01})


def test_mc_missing_cov_columns_raises(market, cov2):
    book = Book([Spot("USDJPY", 1e6)])
    with pytest.raises(ValueError, match="FX:JPY"):
        monte_carlo_var(book, market, cov2, 0.99, n_scenarios=100)


def test_mc_empty_book(market, cov2):
    res = monte_carlo_var(Book([]), market, cov2, 0.99, n_scenarios=100)
    assert res.var == 0.0 and res.es == 0.0
