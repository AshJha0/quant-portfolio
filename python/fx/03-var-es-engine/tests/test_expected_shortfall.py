"""Empirical/closed-form VaR & ES, coherence properties, subadditivity."""

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm, t as student_t

from fx_var import (
    empirical_es,
    empirical_var,
    empirical_var_es,
    normal_es,
    normal_var,
    t_es,
    t_var,
)


# ------------------------------------------------------------ exact quantiles
def test_empirical_var_exact_known_array():
    """100 losses of 1..100: alpha=0.95 -> 5th worst = 96; alpha=0.99 -> 100."""
    pnl = -np.arange(1.0, 101.0)
    assert empirical_var(pnl, 0.95) == 96.0
    assert empirical_var(pnl, 0.99) == 100.0
    assert empirical_es(pnl, 0.95) == pytest.approx(98.0)  # mean(96..100)
    assert empirical_es(pnl, 0.99) == pytest.approx(100.0)


def test_empirical_var_noninteger_tail():
    """n=250, alpha=0.99: ceil(2.5)=3rd worst loss is the VaR."""
    pnl = -np.arange(1.0, 251.0)
    assert empirical_var(pnl, 0.99) == 248.0
    # Acerbi-Tasche ES: (250+249)/250-weighted + fractional share of 248
    es = empirical_es(pnl, 0.99)
    expected = ((250.0 + 249.0) * (1 / 250) + (0.01 - 2 / 250) * 248.0) / 0.01
    assert es == pytest.approx(expected, abs=1e-9)


def test_weighted_var_hand_computed():
    """Weights concentrate on recent big losses: VaR jumps accordingly."""
    pnl = np.array([-10.0, -1.0, -2.0, -3.0])
    w = np.array([0.05, 0.4, 0.4, 0.15])
    # losses desc: 10 (w .05), 3 (w .15), 2, 1; alpha=.9 -> target .1:
    # cum .05 < .1 -> next: VaR = 3
    assert empirical_var(pnl, 0.90, w) == 3.0
    # ES = (10*.05 + 3*(.1-.05))/.1 = (0.5+0.15)/.1 = 6.5
    assert empirical_es(pnl, 0.90, w) == pytest.approx(6.5)


def test_var_es_single_pass_consistent():
    rng = np.random.default_rng(0)
    pnl = rng.standard_normal(5000)
    v, e = empirical_var_es(pnl, 0.975)
    assert v == empirical_var(pnl, 0.975)
    assert e == pytest.approx(empirical_es(pnl, 0.975))


def test_es_geq_var_property():
    rng = np.random.default_rng(1)
    for alpha in (0.9, 0.95, 0.99, 0.999):
        for _ in range(5):
            pnl = rng.standard_t(4, size=750)
            v, e = empirical_var_es(pnl, alpha)
            assert e >= v - 1e-12


def test_input_validation():
    with pytest.raises(ValueError):
        empirical_var([], 0.99)
    with pytest.raises(ValueError):
        empirical_var([1.0, np.nan], 0.99)
    with pytest.raises(ValueError):
        empirical_var([1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        empirical_var([1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        empirical_var([1.0, 2.0], -0.5)
    with pytest.raises(ValueError):
        empirical_var([1.0, 2.0], 0.99, weights=[1.0])
    with pytest.raises(ValueError):
        empirical_var([1.0, 2.0], 0.99, weights=[-1.0, 2.0])
    with pytest.raises(ValueError):
        normal_var(-1.0, 0.99)


# ------------------------------------------------------------ closed forms
def test_normal_es_identity_1e10():
    """ES(alpha) * (1-alpha) == sigma * phi(z_alpha) to 1e-10."""
    for alpha in (0.9, 0.95, 0.975, 0.99):
        sigma = 1.37e6
        es = normal_es(sigma, alpha)
        z = norm.ppf(alpha)
        assert es * (1 - alpha) == pytest.approx(sigma * norm.pdf(z), abs=1e-10 * sigma)


def test_normal_es_vs_numerical_integration():
    """ES = (1/(1-a)) * integral_a^1 VaR(u) du, checked numerically."""
    sigma, alpha = 2.5, 0.99
    val, _ = integrate.quad(lambda u: sigma * norm.ppf(u), alpha, 1.0)
    assert normal_es(sigma, alpha) == pytest.approx(val / (1 - alpha), rel=1e-8)


def test_t_es_vs_numerical_integration():
    sigma, alpha, df = 1.8, 0.975, 5.0
    scale = sigma * np.sqrt((df - 2) / df)
    val, _ = integrate.quad(lambda u: scale * student_t.ppf(u, df), alpha, 1.0)
    assert t_es(sigma, alpha, df) == pytest.approx(val / (1 - alpha), rel=1e-8)


def test_t_fat_tails_vs_normal_at_same_sigma():
    """Variance-matched t has higher 99% VaR/ES than normal - the EM story."""
    sigma = 1.0
    assert t_var(sigma, 0.99, df=4) > normal_var(sigma, 0.99)
    assert t_es(sigma, 0.99, df=4) > normal_es(sigma, 0.99)
    # but a *lower* 95% VaR: fat tails borrow mass from the shoulders
    assert t_var(sigma, 0.95, df=4) < normal_var(sigma, 0.95)


def test_t_requires_df_above_2():
    with pytest.raises(ValueError):
        t_var(1.0, 0.99, df=2.0)
    with pytest.raises(ValueError):
        t_es(1.0, 0.99, df=1.5)


def test_normal_var_mean_shift():
    assert normal_var(1.0, 0.99, mean=0.5) == pytest.approx(norm.ppf(0.99) - 0.5)


# ------------------------------------------------------------ subadditivity
def _joint_peg_assets():
    """Two independent peg-jump assets: loss 10 w.p. 0.9%, else 0.

    Returns per-asset and joint discrete distributions as (pnl, prob).
    """
    p_jump, loss = 0.009, 10.0
    pnl_a = np.array([0.0, -loss])
    w_a = np.array([1 - p_jump, p_jump])
    pnl_sum = np.array([0.0, -loss, -loss, -2 * loss])
    w_sum = np.array([(1 - p_jump) ** 2, (1 - p_jump) * p_jump,
                      p_jump * (1 - p_jump), p_jump**2])
    return (pnl_a, w_a), (pnl_sum, w_sum)


def test_var_not_subadditive_peg_jump():
    """VaR(A+B) > VaR(A) + VaR(B): each peg alone hides under the 99%
    quantile (jump prob 0.9% < 1%) but the pair does not (1.79% > 1%)."""
    (pnl_a, w_a), (pnl_s, w_s) = _joint_peg_assets()
    var_a = empirical_var(pnl_a, 0.99, w_a)
    var_sum = empirical_var(pnl_s, 0.99, w_s)
    assert var_a == 0.0
    assert var_sum == 10.0
    assert var_sum > 2 * var_a  # subadditivity violated


def test_es_subadditive_peg_jump():
    """ES on the same construction is subadditive (coherent measure)."""
    (pnl_a, w_a), (pnl_s, w_s) = _joint_peg_assets()
    es_a = empirical_es(pnl_a, 0.99, w_a)
    es_sum = empirical_es(pnl_s, 0.99, w_s)
    # per-asset coherent ES: (10*0.009 + 0*(0.01-0.009))/0.01 = 9.0
    assert es_a == pytest.approx(9.0)
    assert es_sum <= 2 * es_a + 1e-12
    # and ES sees the peg risk VaR reported as zero
    assert es_a > 0.0


def test_es_subadditive_on_random_samples():
    """Comonotonic-free sanity: ES(A+B) <= ES(A) + ES(B) on heavy-tailed
    simulated samples (same scenarios, so estimator subadditivity applies)."""
    rng = np.random.default_rng(5)
    a = rng.standard_t(3, size=20_000)
    b = 0.5 * a + rng.standard_t(3, size=20_000)
    for alpha in (0.95, 0.99):
        assert empirical_es(a + b, alpha) <= empirical_es(a, alpha) + empirical_es(b, alpha) + 1e-9
