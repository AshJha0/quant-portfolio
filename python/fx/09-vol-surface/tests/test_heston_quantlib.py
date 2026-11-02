"""Optional outside check: Heston prices against QuantLib's AnalyticHestonEngine.

Skipped unless QuantLib is installed (it is not a CI dependency):
``pip install QuantLib``. QuantLib 1.44, flat ACT/365F domestic and foreign
curves, maturities a whole number of days so the year fractions match ours,
engine run with integration tolerance 1e-14. Tolerances are what the
implementations achieve: Gil-Pelaez to 1e-12 relative (about 1e-15 to
5e-14 seen); COS (N=1024, L=24) to 1e-9 relative (about 4e-11 on the
Fang-Oosterlee case, 3e-10 on the T=10, xi=1, rho=-0.9 case).
"""

import pytest
from fx_surface.heston import HestonParams, heston_price

ql = pytest.importorskip("QuantLib")

CASES = [
    # S, K, T, r_d, r_f, v0, kappa, theta, xi, rho
    (100.0, 100.0, 1.0, 0.0, 0.0, 0.0175, 1.5768, 0.0398, 0.5751, -0.5711),  # Fang-Oosterlee
    (1.10, 1.15, 2.0, 0.03, 0.01, 0.01, 2.0, 0.012, 0.3, -0.3),
    (100.0, 120.0, 10.0, 0.02, 0.0, 0.04, 1.0, 0.04, 1.0, -0.9),
    (100.0, 80.0, 0.2, 0.05, 0.02, 0.09, 3.0, 0.06, 0.6, 0.2),
]


def _quantlib(S, K, T, rd, rf, v0, kappa, theta, xi, rho, cp):
    d0 = ql.Date(15, 1, 2026)
    ql.Settings.instance().evaluationDate = d0
    dc = ql.Actual365Fixed()
    days = round(365 * T)
    assert days == 365 * T, "maturity must be a whole number of days"
    proc = ql.HestonProcess(ql.YieldTermStructureHandle(ql.FlatForward(d0, rd, dc)),
                            ql.YieldTermStructureHandle(ql.FlatForward(d0, rf, dc)),
                            ql.QuoteHandle(ql.SimpleQuote(S)), v0, kappa, theta, xi, rho)
    opt = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call if cp == 1 else ql.Option.Put, K),
        ql.EuropeanExercise(d0 + days))
    opt.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(proc), 1e-14, 100000))
    return opt.NPV()


@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize(("S", "K", "T", "rd", "rf", "v0", "kappa", "theta", "xi", "rho"), CASES)
def test_heston_matches_quantlib(S, K, T, rd, rf, v0, kappa, theta, xi, rho, cp):
    q = _quantlib(S, K, T, rd, rf, v0, kappa, theta, xi, rho, cp)
    p = HestonParams(v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho)
    assert heston_price(S, K, T, rd, rf, p, cp, method="gil_pelaez") == pytest.approx(q, rel=1e-12)
    assert heston_price(S, K, T, rd, rf, p, cp, method="cos") == pytest.approx(q, rel=1e-9)
