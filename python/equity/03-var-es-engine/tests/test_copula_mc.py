"""t copula: Kendall tau and tail dependence oracles, Gaussian limit."""

import numpy as np
import pytest
from eq_var.copula_mc import (
    kendall_tau_to_rho,
    t_copula_uniforms,
    t_copula_var_es,
    t_tail_dependence,
)
from scipy.stats import kendalltau, norm

R = np.array([[1.0, 0.5], [0.5, 1.0]])


def test_kendall_tau_oracle():
    u = t_copula_uniforms(R, 4.0, 20_000, seed=1)
    tau = kendalltau(u[:, 0], u[:, 1]).statistic
    assert tau == pytest.approx(2 / np.pi * np.arcsin(0.5), abs=0.01)
    assert kendall_tau_to_rho(2 / np.pi * np.arcsin(0.5)) == pytest.approx(0.5, rel=1e-14)


def test_uniform_marginals():
    u = t_copula_uniforms(R, 3.0, 50_000, seed=2)
    for j in range(2):
        np.testing.assert_allclose(np.quantile(u[:, j], [0.1, 0.5, 0.9]), [0.1, 0.5, 0.9], atol=0.01)


def test_tail_dependence_oracle():
    nu, q = 3.0, 0.005
    u = t_copula_uniforms(R, nu, 2_000_000, seed=3)
    emp = np.mean((u[:, 0] < q) & (u[:, 1] < q)) / q
    lam = t_tail_dependence(0.5, nu)
    assert lam == pytest.approx(0.3125, abs=0.001)  # 2 t_4(-sqrt(4/3)), by hand
    assert emp == pytest.approx(lam, abs=0.06)


def test_more_tail_risk_than_gaussian_limit():
    m = [norm(scale=0.01).ppf, norm(scale=0.01).ppf]
    _, e3 = t_copula_var_es([1.0, 1.0], R, 3.0, m, 0.01, 400_000, seed=4)
    vg, eg = t_copula_var_es([1.0, 1.0], R, 1e6, m, 0.01, 400_000, seed=4)
    exact = 0.01 * np.sqrt(3.0) * norm.ppf(0.99)  # sum of two N(0, .01^2), rho .5
    assert vg == pytest.approx(exact, rel=0.02)
    assert e3 > eg


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        t_copula_uniforms([[1.0, 1.2], [1.2, 1.0]], 4.0, 10)
    with pytest.raises(ValueError):
        t_copula_uniforms(R, 0.0, 10)
    with pytest.raises(ValueError):
        t_tail_dependence(1.0, 4.0)
