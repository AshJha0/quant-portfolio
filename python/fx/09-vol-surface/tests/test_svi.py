"""SVI: fitting, recovery, Durrleman butterfly-arbitrage checks, VV comparison."""

import numpy as np
import pytest

from fx_surface import SVIParams, SVISmile, VannaVolgaSmile
from fx_surface.surface import build_slice

F, T = 1.105, 0.5


def _smile_from_params(params: SVIParams) -> SVISmile:
    return SVISmile(params, F, T)


def test_svi_known_param_recovery():
    true = SVIParams(a=0.0016, b=0.012, rho=-0.35, m=0.02, s=0.08)
    smile = _smile_from_params(true)
    # sample five "pillar-like" strikes
    ks = np.array([-0.12, -0.05, 0.0, 0.05, 0.12])
    strikes = F * np.exp(ks)
    vols = np.asarray(smile.vol(strikes))
    fitted = SVISmile.fit(strikes, vols, F, T)
    # curve recovery on a dense grid (the meaningful test)
    grid = np.linspace(-0.25, 0.25, 101)
    np.testing.assert_allclose(
        fitted.total_variance(grid), smile.total_variance(grid), atol=1e-8
    )
    # parameter recovery (5 points / 5 params -> identified)
    assert fitted.params.a == pytest.approx(true.a, abs=2e-4)
    assert fitted.params.b == pytest.approx(true.b, rel=0.05)
    assert fitted.params.rho == pytest.approx(true.rho, abs=0.05)
    assert fitted.params.m == pytest.approx(true.m, abs=0.02)
    assert fitted.params.s == pytest.approx(true.s, abs=0.02)


def test_svi_fits_eurusd_pillars_exactly(eurusd_surface):
    """Mild G10 smile: 5 points / 5 params -> exact interpolation."""
    for sl in eurusd_surface.slices:
        for p, K in sl.strikes.items():
            assert float(sl.smile.vol(K)) == pytest.approx(sl.vols[p], abs=1e-7)


def test_svi_fits_usdjpy_pillars_to_desk_tolerance(usdjpy_surface):
    """The extreme JPY skew is NOT exactly raw-SVI-interpolable (the
    exact fit would need rho < -1); best fit lands on the rho bound with
    <= ~0.05 vol pts pillar residual - a documented limitation
    (docs/VALIDATION.md)."""
    worst = 0.0
    for sl in usdjpy_surface.slices:
        for p, K in sl.strikes.items():
            worst = max(worst, abs(float(sl.smile.vol(K)) - sl.vols[p]))
    assert worst < 5e-4  # 0.05 vol points


def test_durrleman_nonnegative_on_good_smiles(eurusd_surface, usdjpy_surface):
    for surf in (eurusd_surface, usdjpy_surface):
        for sl in surf.slices:
            ok, g_min = sl.smile.is_butterfly_arbitrage_free()
            assert ok, f"{sl.label}: min g = {g_min}"


def test_planted_butterfly_arbitrage_flagged():
    # Aggressive convexity/skew: total-variance wings too steep for the
    # level -> Durrleman g < 0 somewhere.
    bad = SVIParams(a=0.0004, b=0.35, rho=-0.9, m=0.0, s=0.02)
    smile = _smile_from_params(bad)
    ok, g_min = smile.is_butterfly_arbitrage_free()
    assert not ok
    assert g_min < 0.0


def test_svi_derivatives_match_finite_differences():
    p = SVIParams(a=0.001, b=0.02, rho=-0.4, m=0.01, s=0.1)
    smile = _smile_from_params(p)
    k = np.linspace(-0.3, 0.3, 25)
    h = 1e-6
    wp_fd = (smile.total_variance(k + h) - smile.total_variance(k - h)) / (2 * h)
    wpp_fd = (
        smile.total_variance(k + h) - 2 * smile.total_variance(k) + smile.total_variance(k - h)
    ) / h**2
    np.testing.assert_allclose(smile.w_prime(k), wp_fd, atol=1e-8)
    np.testing.assert_allclose(smile.w_second(k), wpp_fd, atol=1e-3)


def test_svi_vs_vanna_volga_agree_near_pillars(eurusd):
    """SVI and VV are two different interpolators through the same pillar
    quotes; between 25d and ATM they must agree to desk tolerance."""
    ms = eurusd.slices[2]  # 3m
    sl_svi = build_slice(ms.label, ms.T, eurusd.S, ms.r_d, ms.r_f, ms.quotes,
                         ms.convention, "svi")
    sl_vv = build_slice(ms.label, ms.T, eurusd.S, ms.r_d, ms.r_f, ms.quotes,
                        ms.convention, "vv")
    K25p, Katm, K25c = (sl_svi.strikes[p] for p in ("25p", "atm", "25c"))
    for K in np.linspace(K25p, K25c, 21):
        v_svi = float(sl_svi.smile.vol(K))
        v_vv = float(sl_vv.smile.vol(K))
        assert abs(v_svi - v_vv) < 0.0015, f"K={K}: {v_svi} vs {v_vv}"


def test_svi_vs_vv_wings_diverge(eurusd):
    """Beyond the 10d pillars the two constructions genuinely differ
    (VV extrapolates the quadratic vol cost, SVI has linear total
    variance) - the divergence is the documented model risk."""
    ms = eurusd.slices[4]  # 1y
    sl_svi = build_slice(ms.label, ms.T, eurusd.S, ms.r_d, ms.r_f, ms.quotes,
                         ms.convention, "svi")
    sl_vv = build_slice(ms.label, ms.T, eurusd.S, ms.r_d, ms.r_f, ms.quotes,
                        ms.convention, "vv")
    K_far = sl_svi.strikes["10c"] * 1.08
    v_svi = float(sl_svi.smile.vol(K_far))
    v_vv = float(sl_vv.smile.vol(K_far))
    assert abs(v_svi - v_vv) > 1e-4


def test_flat_smile_degenerates_gracefully():
    strikes = F * np.exp(np.array([-0.1, -0.05, 0.0, 0.05, 0.1]))
    vols = np.full(5, 0.08)
    smile = SVISmile.fit(strikes, vols, F, T)
    assert smile.params.b == 0.0  # degenerate branch taken
    grid = np.linspace(-0.5, 0.5, 41)
    np.testing.assert_allclose(smile.vol_logm(grid), 0.08, atol=1e-12)
    ok, _ = smile.is_butterfly_arbitrage_free()
    assert ok


def test_svi_param_validation():
    with pytest.raises(ValueError, match="b must be"):
        SVIParams(a=0.001, b=-0.1, rho=0.0, m=0.0, s=0.1)
    with pytest.raises(ValueError, match="rho"):
        SVIParams(a=0.001, b=0.1, rho=1.0, m=0.0, s=0.1)
    with pytest.raises(ValueError, match="s must be"):
        SVIParams(a=0.001, b=0.1, rho=0.0, m=0.0, s=0.0)
    with pytest.raises(ValueError, match="minimum total variance"):
        SVIParams(a=-0.01, b=0.01, rho=0.0, m=0.0, s=0.1)


def test_svi_fit_input_validation():
    with pytest.raises(ValueError, match="1-D"):
        SVISmile.fit(np.array([1.0, 1.1]), np.array([0.1, 0.1]), F, T)
    with pytest.raises(ValueError, match="positive"):
        SVISmile.fit(np.array([1.0, 1.1, 1.2]), np.array([0.1, -0.1, 0.1]), F, T)
