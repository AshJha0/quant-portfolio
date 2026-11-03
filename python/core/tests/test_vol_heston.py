"""The shared Heston characteristic function (Round 18, W20 Heston merge).

The merge promise is "no number moves": ``qp_core.vol.heston.heston_cf``
must give exactly the bits of both copies it replaced. The two old copies
are kept here verbatim (from ``milestone/v2.1`` at 21709ef) as the oracle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qp_core.vol.heston import heston_cf


def _old_eq_cf(u, T, S, r, q, v0, kappa, theta, xi, rho):
    u = np.asarray(u, dtype=complex)
    x0 = np.log(S) + (r - q) * T
    if xi < 1e-12:
        iv = theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa
        out = np.exp(1j * u * (x0 - 0.5 * iv) - 0.5 * u * u * iv)
        return out if out.ndim else complex(out)
    xi2 = xi * xi
    b = kappa - 1j * rho * xi * u
    d = np.sqrt(b * b + xi2 * (1j * u + u * u))
    g2 = (b - d) / (b + d)
    e_dT = np.exp(-d * T)
    C = (kappa * theta / xi2) * ((b - d) * T - 2.0 * np.log((1.0 - g2 * e_dT) / (1.0 - g2)))
    D = ((b - d) / xi2) * (1.0 - e_dT) / (1.0 - g2 * e_dT)
    out = np.exp(1j * u * x0 + C + D * v0)
    return out if out.ndim else complex(out)


def _old_fx_cf(u, T, v0, kappa, theta, xi, rho, mu=0.0, x0=0.0):
    u = np.asarray(u, dtype=complex)
    beta = kappa - 1j * rho * xi * u
    d = np.sqrt(beta * beta + xi * xi * (1j * u + u * u))
    g = (beta - d) / (beta + d)
    e_dt = np.exp(-d * T)
    D = (beta - d) / (xi * xi) * (1.0 - e_dt) / (1.0 - g * e_dt)
    C = (kappa * theta / (xi * xi)) * ((beta - d) * T - 2.0 * np.log((1.0 - g * e_dt) / (1.0 - g)))
    return np.exp(1j * u * (x0 + mu * T) + C + D * v0)


PARAMS = [
    dict(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.6),
    dict(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711),
    dict(v0=0.04, kappa=1.0, theta=0.04, xi=1.0, rho=-0.9),
    dict(v0=0.09, kappa=3.0, theta=0.06, xi=0.6, rho=0.2),
    dict(v0=0.01, kappa=2.0, theta=0.012, xi=1e-8, rho=-1.0),
]
U = np.concatenate([np.linspace(0.0, 200.0, 41), np.linspace(1.0, 50.0, 7) - 1j,
                    np.linspace(0.0, 80.0, 9) - 2.5j])


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.array_equal(a.view(np.float64), b.view(np.float64), equal_nan=True)


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("T", [0.0, 0.02, 1.0, 10.0])
@pytest.mark.parametrize("S,r,q", [(100.0, 0.05, 0.02), (1.3, 0.03, 0.01), (100.0, 0.0, 0.0)])
def test_equals_old_equity_copy_bit_for_bit(p, T, S, r, q):
    new = heston_cf(U, T, **p, x0=np.log(S), mu=r - q)
    assert _same(new, _old_eq_cf(U, T, S, r, q, **p))


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("T", [0.02, 1.0, 10.0])
@pytest.mark.parametrize("x0,mu", [(0.0, 0.0), (math.log(1.3), 0.02), (0.0, -0.01)])
def test_equals_old_fx_copy_bit_for_bit(p, T, x0, mu):
    new = heston_cf(U, T, **p, x0=x0, mu=mu)
    assert _same(new, _old_fx_cf(U, T, **p, mu=mu, x0=x0))


def test_deterministic_variance_branch_equals_old_equity_copy():
    p = dict(v0=0.04, kappa=1.5, theta=0.05, xi=0.0, rho=-0.6)
    new = heston_cf(U, 1.0, **p, x0=np.log(100.0), mu=0.03)
    assert _same(new, _old_eq_cf(U, 1.0, 100.0, 0.05, 0.02, **p))


def test_martingale_and_normalisation():
    p = PARAMS[0]
    assert heston_cf(0.0, 2.0, **p, x0=1.0, mu=0.1) == 1.0
    fwd = heston_cf(-1j, 2.0, **p, x0=math.log(100.0), mu=0.03)
    assert isinstance(fwd, complex)
    assert fwd.real == pytest.approx(100.0 * math.exp(0.06), rel=1e-13)
    assert abs(fwd.imag) < 1e-12


def test_negative_maturity_raises():
    with pytest.raises(ValueError, match="non-negative"):
        heston_cf(1.0, -0.1, **PARAMS[0])


QL_CASES = [
    # S, K, T, r, q, v0, kappa, theta, xi, rho; T a whole number of days
    (100.0, 100.0, 1.0, 0.0, 0.0, 0.0175, 1.5768, 0.0398, 0.5751, -0.5711),
    (1.10, 1.15, 2.0, 0.03, 0.01, 0.01, 2.0, 0.012, 0.3, -0.3),
    (100.0, 80.0, 0.2, 0.05, 0.02, 0.09, 3.0, 0.06, 0.6, 0.2),
]


@pytest.mark.parametrize("case", QL_CASES)
def test_gil_pelaez_on_shared_cf_matches_quantlib(case):
    """Outside oracle: QuantLib 1.44 AnalyticHestonEngine (tolerance 1e-14).

    The call is priced from the shared characteristic function alone, by
    Gil-Pelaez inversion with adaptive quadrature, so this checks the CF
    itself rather than either package's pricer. Optional: skipped without
    QuantLib (not a CI dependency).
    """
    ql = pytest.importorskip("QuantLib")
    from scipy.integrate import quad

    S, K, T, r, q, v0, kappa, theta, xi, rho = case
    p = dict(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    x0, mu, lnK = math.log(S), r - q, math.log(K)
    F = S * math.exp(mu * T)

    def integrand(u, shift):
        cf = heston_cf(u - shift, T, **p, x0=x0, mu=mu)
        return (np.exp(-1j * u * lnK) * cf / (1j * u)).real

    i1 = quad(lambda u: integrand(u, 1j) / F, 0.0, np.inf, limit=250, epsabs=1e-13)[0]
    i2 = quad(lambda u: integrand(u, 0.0), 0.0, np.inf, limit=250, epsabs=1e-13)[0]
    ours = math.exp(-r * T) * (F * (0.5 + i1 / math.pi) - K * (0.5 + i2 / math.pi))

    d0 = ql.Date(15, 1, 2026)
    ql.Settings.instance().evaluationDate = d0
    dc = ql.Actual365Fixed()

    def curve(rate):
        return ql.YieldTermStructureHandle(ql.FlatForward(d0, rate, dc))

    proc = ql.HestonProcess(curve(r), curve(q), ql.QuoteHandle(ql.SimpleQuote(S)),
                            v0, kappa, theta, xi, rho)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K),
                           ql.EuropeanExercise(d0 + round(365 * T)))
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc), 1e-14, 100000))
    assert ours == pytest.approx(opt.NPV(), rel=1e-11, abs=1e-13)
