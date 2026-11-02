"""Phase 5 equity models, each against an oracle.

Oracles: closed forms (Black-Scholes, Roll-Geske-Whaley with Haug's
published value, Merton term-structure averages), mpmath at 30 digits
(higher Greeks, Bachelier integral), and convergence (LR order 2, CN order
2, ALO against a Richardson-extrapolated LR tree).
"""

import math

import mpmath as mp
import numpy as np
import pytest
from eq_options import bs_greeks, bs_price
from eq_options.alo import alo_price
from eq_options.bachelier import bachelier_greeks, bachelier_implied_vol, bachelier_price
from eq_options.dividends import bs_price_discrete_divs, lr_price_discrete_divs, rgw_call
from eq_options.greeks_higher import higher_greeks
from eq_options.lr_tree import lr_price, peizer_pratt
from eq_options.pde import cn_price
from eq_options.risk_grids import book_value, scenario_grid, spot_ladder, vega_ladder
from eq_options.sabr import sabr_density_check, sabr_vol
from eq_options.term_structure import VolTermStructure, ZeroCurve, bs_price_ts, flat_curve

CASES = [
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),
    (100.0, 110.0, 0.5, 0.03, 0.30, 0.02),
    (100.0, 80.0, 2.0, 0.01, 0.15, 0.04),
    (50.0, 60.0, 0.25, -0.01, 0.40, 0.00),
]


def _lr_richardson(*a, n=2001):
    return 2.0 * lr_price(*a, "american", 2 * n - 1) - lr_price(*a, "american", n)


# ------------------------------------------------------------- LR tree ----
@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
@pytest.mark.parametrize("ot", ["call", "put"])
def test_lr_european_matches_bs(S, K, T, r, sigma, q, ot):
    assert lr_price(S, K, T, r, sigma, q, ot, "european", 401) == pytest.approx(
        bs_price(S, K, T, r, sigma, q, ot), abs=2e-5)


def test_lr_second_order_convergence():
    b = bs_price(100, 105, 1, 0.04, 0.25, 0.01, "call")
    errs = [abs(lr_price(100, 105, 1, 0.04, 0.25, 0.01, "call", "european", n) - b)
            for n in (101, 201, 401)]
    for e1, e2 in zip(errs, errs[1:], strict=True):
        assert 3.0 < e1 / e2 < 5.0


def test_peizer_pratt_symmetric_and_rejects_even():
    assert peizer_pratt(0.0, 11) == 0.5
    assert peizer_pratt(0.7, 11) + peizer_pratt(-0.7, 11) == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError):
        peizer_pratt(0.1, 10)


def test_lr_american_call_no_dividend_equals_european():
    a = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    assert lr_price(*a, "american", 301) == pytest.approx(lr_price(*a, "european", 301), abs=1e-12)


# ---------------------------------------------------------- CN PDE --------
def test_cn_european_matches_bs():
    for ot in ("call", "put"):
        assert cn_price(100, 100, 1, 0.05, 0.2, 0.01, ot, "european", 800, 400) == pytest.approx(
            bs_price(100, 100, 1, 0.05, 0.2, 0.01, ot), abs=5e-4)


def test_cn_american_second_order_and_matches_tree():
    a = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "put")
    ref = _lr_richardson(*a)
    e = [abs(cn_price(*a, "american", n, n) - ref) for n in (200, 400, 800)]
    assert e[2] < 3e-4
    assert e[0] / e[1] > 3.0 and e[1] / e[2] > 3.0


# ------------------------------------------------------------- ALO -------
@pytest.mark.parametrize("a", [
    (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "put"),
    (100.0, 90.0, 0.5, 0.03, 0.3, 0.06, "call"),
    (100.0, 100.0, 1.0, 0.02, 0.2, 0.06, "put"),
    (110.0, 100.0, 2.0, 0.06, 0.25, 0.02, "put"),
])
def test_alo_matches_extrapolated_tree(a):
    assert alo_price(*a) == pytest.approx(_lr_richardson(*a), abs=2e-5)


def test_alo_no_early_exercise_regions_are_european():
    assert alo_price(100, 100, 1, 0.05, 0.2, 0.0, "call") == bs_price(100, 100, 1, 0.05, 0.2, 0.0, "call")
    assert alo_price(100, 100, 1, -0.01, 0.2, 0.0, "put") == bs_price(100, 100, 1, -0.01, 0.2, 0.0, "put")


def test_alo_deep_itm_put_is_intrinsic():
    assert alo_price(40, 100, 1, 0.08, 0.2, 0.0, "put") == 60.0


# ------------------------------------------------------ dividends ---------
def test_rgw_haug_published_value():
    # Haug (2007), section 9.2.1 example: 4.3860.
    assert rgw_call(80, 82, 0.25, 1 / 3, 4, 0.06, 0.30) == pytest.approx(4.3860, abs=5e-5)


def test_dividend_tree_converges_to_rgw():
    rgw = rgw_call(80, 82, 0.25, 1 / 3, 4, 0.06, 0.30)
    assert lr_price_discrete_divs(80, 82, 1 / 3, 0.06, 0.30, [(0.25, 4.0)], "call",
                                  "american", 4001) == pytest.approx(rgw, abs=1e-3)


def test_rgw_small_dividend_falls_back_to_escrowed_european():
    v = rgw_call(100, 100, 0.5, 1.0, 0.1, 0.05, 0.2)
    assert v == bs_price_discrete_divs(100, 100, 1.0, 0.05, 0.2, [(0.5, 0.1)])


def test_escrowed_european_tree_matches_closed_form():
    divs = [(0.2, 1.0), (0.7, 1.5), (1.5, 9.0)]  # last one after expiry: ignored
    for ot in ("call", "put"):
        assert lr_price_discrete_divs(100, 95, 1.0, 0.04, 0.25, divs, ot, "european", 801) == \
            pytest.approx(bs_price_discrete_divs(100, 95, 1.0, 0.04, 0.25, divs, ot), abs=1e-5)


# --------------------------------------------------- term structures ------
def test_flat_term_structure_is_bs():
    v = bs_price_ts(100, 105, 1.3, flat_curve(0.03), VolTermStructure((1.0,), (0.25,)),
                    flat_curve(0.01), "put")
    assert v == pytest.approx(bs_price(100, 105, 1.3, 0.03, 0.25, 0.01, "put"), rel=1e-14)


def test_term_structure_worked_example_and_merton_average():
    z = ZeroCurve((1.0, 2.0), (0.02, 0.04))
    assert z.forward_rate(1.0, 2.0) == pytest.approx(0.06, abs=1e-15)
    vt = VolTermStructure((1.0, 2.0), (0.2, 0.3))
    assert vt.forward_vol(1.0, 2.0) == pytest.approx(math.sqrt(0.14), abs=1e-15)
    # 1.5y: integral of forward = 0.02 + 0.06*0.5; variance = 0.04 + 0.14*0.5.
    v = bs_price_ts(100, 100, 1.5, z, vt)
    assert v == pytest.approx(bs_price(100, 100, 1.5, 0.05 / 1.5, math.sqrt(0.11 / 1.5)), rel=1e-14)


def test_calendar_arbitrage_rejected():
    with pytest.raises(ValueError):
        VolTermStructure((1.0, 2.0), (0.3, 0.2))


# ------------------------------------------------------------ Bachelier ---
def test_bachelier_matches_mpmath_integral():
    mp.mp.dps = 30
    F, K, T, s, r = 1.0, 1.3, 2.0, 0.4, 0.03
    sd = s * mp.sqrt(T)
    dens = lambda x: mp.npdf(x, F, sd)  # noqa: E731
    call = mp.exp(-r * T) * mp.quad(lambda x: (x - K) * dens(x), [K, mp.inf])
    put = mp.exp(-r * T) * mp.quad(lambda x: (K - x) * dens(x), [-mp.inf, K])
    assert bachelier_price(F, K, T, s, r, "call") == pytest.approx(float(call), abs=1e-15)
    assert bachelier_price(F, K, T, s, r, "put") == pytest.approx(float(put), abs=1e-15)


def test_bachelier_worked_example_negative_forward_and_parity():
    assert bachelier_price(100, 100, 1, 20) == pytest.approx(20 / math.sqrt(2 * math.pi), rel=1e-15)
    c = bachelier_price(-0.5, 0.2, 1, 0.8, 0.01, "call")
    p = bachelier_price(-0.5, 0.2, 1, 0.8, 0.01, "put")
    assert c - p == pytest.approx(math.exp(-0.01) * (-0.7), abs=1e-15)


def test_bachelier_greeks_vs_mpmath():
    mp.mp.dps = 30
    F, K, T, s, r = 1.0, 1.1, 0.7, 0.3, 0.02

    def px(f=F, ss=s, t=T):
        d = (f - K) / (ss * mp.sqrt(t))
        return mp.exp(-r * t) * ((f - K) * mp.ncdf(d) + ss * mp.sqrt(t) * mp.npdf(d))

    g = bachelier_greeks(F, K, T, s, r, "call")
    assert g.delta == pytest.approx(float(mp.diff(lambda f: px(f=f), F)), abs=1e-13)
    assert g.gamma == pytest.approx(float(mp.diff(lambda f: px(f=f), F, 2)), abs=1e-12)
    assert g.vega == pytest.approx(float(mp.diff(lambda v: px(ss=v), s)), abs=1e-13)
    assert g.theta == pytest.approx(-float(mp.diff(lambda t: px(t=t), T)), abs=1e-13)


@pytest.mark.parametrize("K", [-1.0, 0.5, 1.0, 1.5, 4.0])
def test_bachelier_implied_vol_round_trip(K):
    for ot in ("call", "put"):
        p = bachelier_price(1.0, K, 1.5, 0.6, 0.02, ot)
        assert bachelier_implied_vol(p, 1.0, K, 1.5, 0.02, ot) == pytest.approx(0.6, rel=1e-10)
    with pytest.raises(ValueError):
        bachelier_implied_vol(0.0, 1.0, 1.0, 1.0)


# -------------------------------------------------------------- SABR ------
def test_sabr_beta1_nu0_is_flat_alpha():
    for k in (0.5, 1.0, 2.0):
        assert sabr_vol(1.0, k, 3.0, 0.25, 1.0, 0.0, 0.0) == pytest.approx(0.25, rel=1e-15)


def test_sabr_atm_matches_hagan_atm_formula():
    F, T, a, b, rho, nu = 0.03, 5.0, 0.02, 0.5, -0.3, 0.4
    atm = a / F ** (1 - b) * (1 + T * ((1 - b) ** 2 * a * a / (24 * F ** (2 - 2 * b))
                                       + rho * b * nu * a / (4 * F ** (1 - b))
                                       + (2 - 3 * rho * rho) * nu * nu / 24))
    for variant in ("hagan", "obloj"):
        assert sabr_vol(F, F, T, a, b, rho, nu, variant=variant) == pytest.approx(atm, rel=1e-14)


def test_sabr_smooth_through_the_money():
    # The series branch (|z| < 1e-6) and the closed branch meet without a
    # jump: one-sided slopes across the money agree.
    f = lambda k: sabr_vol(0.03, k, 5.0, 0.02, 0.5, -0.3, 0.4)  # noqa: E731
    atm = f(0.03)
    for h in (1e-9, 1e-8):
        up, dn = (f(0.03 + h) - atm) / h, (atm - f(0.03 - h)) / h
        assert up == pytest.approx(dn, rel=1e-5)
    far = (f(0.03 + 1e-5) - f(0.03 - 1e-5)) / 2e-5
    assert (f(0.03 + 1e-9) - f(0.03 - 1e-9)) / 2e-9 == pytest.approx(far, rel=1e-5)


def test_sabr_obloj_and_hagan_agree_for_short_expiry_near_money():
    for k in (0.028, 0.032):
        h = sabr_vol(0.03, k, 0.1, 0.02, 0.5, -0.3, 0.4, variant="hagan")
        o = sabr_vol(0.03, k, 0.1, 0.02, 0.5, -0.3, 0.4, variant="obloj")
        assert h == pytest.approx(o, rel=1e-3)


def test_sabr_density_flags_low_strike_arbitrage():
    k, d = sabr_density_check(0.03, 10.0, 0.02, 0.5, -0.3, 0.6, np.linspace(0.0005, 0.1, 200))
    assert d.min() < 0.0 and k[d < 0].max() < 0.03
    _, d_short = sabr_density_check(0.03, 0.5, 0.02, 0.5, -0.3, 0.3, np.linspace(0.01, 0.06, 101))
    assert d_short.min() > 0.0


def test_shifted_sabr_negative_forward():
    assert sabr_vol(-0.002, 0.001, 1.0, 0.01, 0.5, 0.0, 0.2, shift=0.03) > 0.0


# ------------------------------------------------------ higher Greeks -----
def _mp_bs(S, K, T, r, s, q, ot):
    d1 = (mp.log(S / K) + (r - q + s * s / 2) * T) / (s * mp.sqrt(T))
    d2 = d1 - s * mp.sqrt(T)
    if ot == "call":
        return S * mp.exp(-q * T) * mp.ncdf(d1) - K * mp.exp(-r * T) * mp.ncdf(d2)
    return K * mp.exp(-r * T) * mp.ncdf(-d2) - S * mp.exp(-q * T) * mp.ncdf(-d1)


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
@pytest.mark.parametrize("ot", ["call", "put"])
def test_higher_greeks_vs_mpmath(S, K, T, r, sigma, q, ot):
    mp.mp.dps = 30
    f = lambda s_, t_, v_: _mp_bs(mp.mpf(s_), K, mp.mpf(t_), r, mp.mpf(v_), q, ot)  # noqa: E731
    g = higher_greeks(S, K, T, r, sigma, q, ot)
    ref = {
        "charm": -mp.diff(lambda s_, t_: f(s_, t_, sigma), (S, T), (1, 1)),
        "speed": mp.diff(lambda s_: f(s_, T, sigma), S, 3),
        "colour": -mp.diff(lambda s_, t_: f(s_, t_, sigma), (S, T), (2, 1)),
        "zomma": mp.diff(lambda s_, v_: f(s_, T, v_), (S, sigma), (2, 1)),
        "veta": -mp.diff(lambda t_, v_: f(S, t_, v_), (T, sigma), (1, 1)),
    }
    for name, val in ref.items():
        assert getattr(g, name) == pytest.approx(float(val), rel=1e-9, abs=1e-12), name


# ---------------------------------------------------- ladders and grids ---
BOOK = [(10.0, 100.0, 0.3, "call"), (-5.0, 95.0, 0.8, "put"), (3.0, 110.0, 1.7, "call"),
        (2.0, 100.0, 7.0, "put")]


def test_spot_ladder_matches_direct_pricing():
    lad = spot_ladder(BOOK, 100.0, 0.03, 0.22, 0.01)
    for s, v, d in zip(lad["spot"], lad["value"], lad["delta"], strict=True):
        assert v == pytest.approx(book_value(BOOK, s, 0.03, 0.22, 0.01), rel=1e-14)
        assert d == pytest.approx(sum(n * bs_greeks(s, k, t, 0.03, 0.22, 0.01, o).delta
                                      for n, k, t, o in BOOK), rel=1e-14)


def test_vega_ladder_sums_and_splits():
    lad = vega_ladder(BOOK, 100.0, 0.03, 0.22, 0.01)
    total = sum(n * bs_greeks(100.0, k, t, 0.03, 0.22, 0.01, o).vega for n, k, t, o in BOOK)
    assert sum(lad.values()) == pytest.approx(total, rel=1e-14)
    v03 = 10.0 * bs_greeks(100.0, 100.0, 0.3, 0.03, 0.22, 0.01, "call").vega
    v08 = -5.0 * bs_greeks(100.0, 95.0, 0.8, 0.03, 0.22, 0.01, "put").vega
    assert lad[0.25] == pytest.approx(v03 * 0.8, rel=1e-14)
    assert lad[0.5] == pytest.approx(v03 * 0.2 + v08 * 0.4, rel=1e-14)


def test_scenario_grid_full_revaluation():
    g = scenario_grid(BOOK, 100.0, 0.03, 0.22, 0.01)
    assert g[2, 1] == 0.0
    base = book_value(BOOK, 100.0, 0.03, 0.22, 0.01)
    assert g[0, 2] == pytest.approx(book_value(BOOK, 90.0, 0.03, 0.27, 0.01) - base, rel=1e-14)
