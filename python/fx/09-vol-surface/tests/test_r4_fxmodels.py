"""Round 4: FX Dupire local vol, SSVI/eSSVI, Bates.

Oracles: closed forms (flat vol, term-structure-only, Heston, Merton),
two independent Dupire routes, Monte Carlo repricing, and the equity twins
(``eq_surface``) with ``q = r_f`` where the maths coincides.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fx_surface import HestonParams, gk_delta, gk_forward, gk_price, heston_price
from fx_surface.bates import JumpParams, bates_call, bates_cf_x, bates_put, merton_call
from fx_surface.data import eurusd_market, usdjpy_market
from fx_surface.dupire import (
    LocalVolArbitrageError,
    local_vol_at_delta,
    local_vol_from_prices,
    local_vol_from_total_variance,
    local_vol_mc_price,
)
from fx_surface.smile import SVISmile
from fx_surface.smile_from_quotes import solve_pillar_strikes, vols_from_quotes
from fx_surface.ssvi import (
    ESSVIParams,
    SSVIParams,
    calendar_arbitrage_free,
    fit_ssvi,
    fit_ssvi_from_pillars,
    ssvi_slice_as_svi,
    ssvi_strike_at_delta,
    ssvi_total_variance,
)

K_GRID = np.linspace(-1.0, 1.0, 41)
# EURUSD-like: domestic USD, foreign EUR
S, RD, RF = 1.10, 0.045, 0.035
P = SSVIParams(rho=-0.3, eta=0.9, gamma=0.4)


def _w(k, T):
    return P.total_variance(k, 0.0064 * T)


def _eq(module: str):
    """Import an equity twin module from the repo for cross-checks."""
    src = Path(__file__).resolve().parents[3] / "equity" / "09-vol-surface" / "src"
    if not src.is_dir():
        pytest.skip("equity package not in this checkout")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return pytest.importorskip(f"eq_surface.{module}")


# ----------------------------------------------------------------- SSVI
def test_ssvi_slice_equals_raw_svi():
    theta, rho, phi = 0.01, -0.4, 4.0
    svi = SVISmile(ssvi_slice_as_svi(theta, rho, phi), F=1.1, T=1.0)
    np.testing.assert_allclose(svi.total_variance(K_GRID),
                               ssvi_total_variance(K_GRID, theta, rho, phi), rtol=1e-13)


def test_ssvi_matches_equity_twin():
    eq = _eq("ssvi")
    pe = eq.SSVIParams(rho=-0.3, eta=0.9, gamma=0.4)
    for th in (0.002, 0.01, 0.05):
        np.testing.assert_allclose(P.total_variance(K_GRID, th), pe.total_variance(K_GRID, th),
                                   rtol=0, atol=1e-16)


def test_ssvi_conditions_and_calendar():
    assert P.static_arbitrage_free()
    assert all(P.butterfly_free(th) for th in (0.001, 0.01, 0.1))
    assert calendar_arbitrage_free(P, np.array([0.002, 0.005, 0.01]))
    assert not calendar_arbitrage_free(P, np.array([0.01, 0.005]))
    with pytest.raises(ValueError):
        SSVIParams(rho=1.0, eta=1.0, gamma=0.3)
    with pytest.raises(ValueError):
        SSVIParams(rho=0.0, eta=float("nan"), gamma=0.3)
    with pytest.raises(ValueError):
        ESSVIParams(-0.2, -0.1, 0.02, 0.01, 1.0, 0.3)


def test_fit_ssvi_recovers_params_and_essvi_rho():
    thetas = np.array([0.0016, 0.0064, 0.0144])
    k = [K_GRID] * 3
    w = [P.total_variance(K_GRID, t) for t in thetas]
    res = fit_ssvi(k, w, thetas)
    assert res.rmse_total_variance < 1e-8
    assert res.params.rho == pytest.approx(P.rho, abs=1e-5)
    pe = ESSVIParams(-0.1, -0.4, thetas[0], thetas[-1], 0.8, 0.4)
    we = [pe.total_variance(K_GRID, t) for t in thetas]
    rese = fit_ssvi(k, we, thetas, extended=True)
    assert rese.rmse_total_variance < 1e-7
    with pytest.raises(ValueError):
        fit_ssvi(k, w[:2], thetas)
    with pytest.raises(ValueError):
        fit_ssvi(k, w, np.array([0.0, 0.1, 0.2]))


def test_fit_from_pillars_reads_theta_at_forward_not_dns():
    """FX DNS ATM sits off the forward; theta is the k=0 total variance."""
    mkt = usdjpy_market()
    strikes, vols, fwds, Ts = [], [], [], []
    for sl in mkt.slices[:4]:
        v = vols_from_quotes(sl.quotes)
        K = solve_pillar_strikes(v, mkt.S, sl.T, sl.r_d, sl.r_f, sl.convention)
        labels = ["10p", "25p", "atm", "25c", "10c"]
        strikes.append([K[x] for x in labels])
        vols.append([v[x] for x in labels])
        fwds.append(gk_forward(mkt.S, sl.T, sl.r_d, sl.r_f))
        Ts.append(sl.T)
    res = fit_ssvi_from_pillars(strikes, vols, fwds, Ts)
    # pa DNS: K_atm = F exp(-sigma^2 T/2) < F, so theta != atm^2 T
    for th, v, T in zip(res.thetas, vols, Ts, strict=True):
        assert th != pytest.approx(v[2] ** 2 * T, rel=1e-6)
        assert 0.5 * v[2] ** 2 * T < th < 2.0 * v[2] ** 2 * T
    # Finding, not a tuning target: three-parameter SSVI cannot match the
    # five-point USDJPY smile; the worst pillar misses by about one vol point.
    errs = [np.max(np.abs(np.sqrt(res.params.total_variance(np.log(np.array(K) / F), th) / T)
                          - np.array(v)))
            for K, v, F, T, th in zip(strikes, vols, fwds, Ts, res.thetas, strict=True)]
    assert max(errs) < 0.012
    assert calendar_arbitrage_free(res.params, res.thetas)
    with pytest.raises(ValueError, match="bracket"):
        fit_ssvi_from_pillars([[1.2, 1.3, 1.4]], [[0.1, 0.1, 0.1]], [1.1], [1.0])
    with pytest.raises(ValueError, match="three"):
        fit_ssvi_from_pillars([[1.0, 1.2]], [[0.1, 0.1]], [1.1], [1.0])


@pytest.mark.parametrize("conv", ["spot", "forward", "spot_pa", "forward_pa"])
@pytest.mark.parametrize("cp", [1, -1])
def test_ssvi_strike_at_delta_is_smile_consistent(conv, cp):
    T = 0.5
    theta = 0.0064 * T
    K, vol = ssvi_strike_at_delta(P, theta, 0.25, cp, S, T, RD, RF, conv)
    assert abs(gk_delta(S, K, T, RD, RF, vol, cp, conv)) == pytest.approx(0.25, abs=1e-10)
    F = gk_forward(S, T, RD, RF)
    assert vol == pytest.approx(np.sqrt(P.total_variance(np.log(K / F), theta) / T), rel=1e-12)


# --------------------------------------------------------------- Dupire
def test_dupire_flat_and_term_structure():
    lv = local_vol_from_total_variance(lambda k, T: 0.01 * T + 0 * k, K_GRID, 1.0)
    np.testing.assert_allclose(lv, 0.1, rtol=1e-6)
    a, b = 0.006, 0.001
    lv = local_vol_from_total_variance(lambda k, T: a * T + b * T * T + 0 * k, K_GRID, 2.0)
    np.testing.assert_allclose(lv, np.sqrt(a + 4 * b), rtol=1e-6)


def _gk_call_from_w(K, t):
    F = gk_forward(S, t, RD, RF)
    k = np.log(K / F)
    return gk_price(S, K, t, RD, RF, float(np.sqrt(_w(np.array([k]), t)[0] / t)), 1)


def test_dupire_two_routes_agree_with_fx_carry():
    T = 1.0
    F = gk_forward(S, T, RD, RF)
    for K in (0.95, 1.10, 1.25):
        a = local_vol_from_total_variance(_w, np.log(K / F), T)
        b = local_vol_from_prices(_gk_call_from_w, K, T, RD, RF)
        assert a == pytest.approx(b, rel=2e-4)


def test_dupire_matches_equity_twin_with_q_equal_rf():
    eq = _eq("dupire")
    T = 0.75
    np.testing.assert_allclose(local_vol_from_total_variance(_w, K_GRID, T),
                               eq.local_vol_from_total_variance(_w, K_GRID, T), rtol=1e-15)
    for K in (1.0, 1.15):
        assert local_vol_from_prices(_gk_call_from_w, K, T, RD, RF) == pytest.approx(
            eq.local_vol_from_prices(_gk_call_from_w, K, T, RD, RF), rel=1e-14)


@pytest.mark.parametrize("conv", ["spot", "forward_pa"])
def test_local_vol_at_delta(conv):
    T = 0.5
    K, lv = local_vol_at_delta(_w, 0.25, T, S, RD, RF, cp=-1, convention=conv)
    F = gk_forward(S, T, RD, RF)
    sig = float(np.sqrt(_w(np.array([np.log(K / F)]), T)[0] / T))
    assert abs(gk_delta(S, K, T, RD, RF, sig, -1, conv)) == pytest.approx(0.25, abs=1e-10)
    assert lv == pytest.approx(local_vol_from_total_variance(_w, np.log(K / F), T), rel=1e-14)
    assert K < F  # 25-delta put strike is below the forward


def test_dupire_guards():
    with pytest.raises(LocalVolArbitrageError):
        local_vol_from_total_variance(lambda k, T: 0.01 * (2.0 - T) + 0 * k, 0.0, 1.0)
    with pytest.raises(ValueError):
        local_vol_from_total_variance(_w, 0.0, 0.0)
    with pytest.raises(ValueError):
        local_vol_from_total_variance(_w, float("nan"), 1.0)
    with pytest.raises(ValueError):
        local_vol_from_prices(_gk_call_from_w, -1.0, 1.0, RD, RF)


def test_dupire_mc_reprices_fx_surface():
    T = 0.5
    K = np.array([1.02, 1.10, 1.18])
    px, se = local_vol_mc_price(_w, S, K, T, RD, RF, n_paths=40_000, n_steps=100, seed=11)
    for Ki, p, s in zip(K, px, se, strict=True):
        ref = _gk_call_from_w(Ki, T)
        assert abs(p - ref) < 4.0 * s + 2e-4, (Ki, p, ref, s)


# ---------------------------------------------------------------- Bates
HP = HestonParams(v0=0.01, kappa=1.5, theta=0.012, xi=0.3, rho=-0.3)
JP = JumpParams(lam=0.5, mu_j=-0.03, delta=0.05)


def test_bates_without_jumps_is_fx_heston():
    j0 = JumpParams(lam=0.0, mu_j=-0.1, delta=0.1)
    for K in (1.0, 1.1, 1.2):
        assert bates_call(S, K, 1.0, RD, RF, HP, j0) == pytest.approx(
            heston_price(S, K, 1.0, RD, RF, HP, 1, method="gil_pelaez"), abs=1e-9)


def test_bates_tiny_volvol_is_merton():
    p = HestonParams(v0=0.01, kappa=1.0, theta=0.01, xi=1e-8, rho=0.0)
    for K in (0.95, 1.1, 1.3):
        for T in (0.25, 2.0):
            assert bates_call(S, K, T, RD, RF, p, JP) == pytest.approx(
                merton_call(S, K, T, RD, RF, 0.1, JP), abs=1e-9)


def test_bates_parity_forward_and_cf_martingale():
    T = 1.5
    c = bates_call(S, 1.1, T, RD, RF, HP, JP)
    p = bates_put(S, 1.1, T, RD, RF, HP, JP)
    assert c - p == pytest.approx(S * np.exp(-RF * T) - 1.1 * np.exp(-RD * T), abs=1e-14)
    # E[exp(X)] = 1: the jump compensator keeps the FX forward a martingale
    assert complex(bates_cf_x(-1j, T, HP, JP)) == pytest.approx(1.0, abs=1e-12)


def test_bates_matches_equity_twin_with_q_equal_rf():
    eqb = _eq("bates")
    eqh = _eq("heston")
    he = eqh.HestonParams(v0=HP.v0, kappa=HP.kappa, theta=HP.theta, xi=HP.xi, rho=HP.rho)
    je = eqb.JumpParams(lam=JP.lam, mu_j=JP.mu_j, delta=JP.delta)
    for K in (1.0, 1.1, 1.25):
        assert bates_call(S, K, 1.0, RD, RF, HP, JP) == pytest.approx(
            eqb.bates_call(S, K, 1.0, RD, RF, he, je), abs=1e-12)
    assert merton_call(S, 1.1, 1.0, RD, RF, 0.1, JP) == pytest.approx(
        eqb.merton_call(S, 1.1, 1.0, RD, RF, 0.1, je), rel=1e-13)


def test_bates_guards():
    with pytest.raises(ValueError):
        JumpParams(lam=-1.0, mu_j=0.0, delta=0.1)
    with pytest.raises(ValueError):
        JumpParams(lam=1.0, mu_j=float("nan"), delta=0.1)
    with pytest.raises(ValueError):
        bates_call(S, 1.1, 0.0, RD, RF, HP, JP)
    with pytest.raises(ValueError):
        bates_call(S, 1.1, 1.0, float("inf"), RF, HP, JP)


def test_bates_jumps_fatten_short_dated_wings():
    """Jumps lift short-dated OTM put prices over Heston (the reason Bates exists)."""
    j0 = JumpParams(lam=0.0, mu_j=0.0, delta=0.0)
    T, K = 0.1, 1.02
    assert bates_put(S, K, T, RD, RF, HP, JP) > bates_put(S, K, T, RD, RF, HP, j0)
    mkt = eurusd_market()
    assert mkt.pair == "EURUSD"  # the presets stay importable alongside the new models
