"""Synthetic market generators: determinism, presets, ground-truth mode."""

import numpy as np
import pytest

from fx_surface import HestonParams, implied_vol, price_cos
from fx_surface.data import (
    calibration_slices,
    em_high_vol_market,
    eurusd_market,
    market_from_heston,
    usdjpy_market,
)
from fx_surface.data.synthetic import TENORS


def test_deterministic_given_seed():
    a = eurusd_market(noise=0.001, seed=42)
    b = eurusd_market(noise=0.001, seed=42)
    c = eurusd_market(noise=0.001, seed=43)
    for sa, sb in zip(a.slices, b.slices):
        assert sa.quotes == sb.quotes
    assert any(sa.quotes != sc.quotes for sa, sc in zip(a.slices, c.slices))


def test_noise_zero_is_exact_preset():
    assert eurusd_market(noise=0.0, seed=1).slices == eurusd_market(noise=0.0, seed=2).slices


def test_eurusd_preset_shape():
    m = eurusd_market()
    assert m.pair == "EURUSD" and len(m.slices) == 6
    assert m.tenor_labels == [t[0] for t in TENORS]
    for sl in m.slices:
        assert -0.01 < sl.quotes.rr25 < 0.0  # small negative RR
        assert 0.0 < sl.quotes.bf25 < 0.005  # moderate BF
        assert sl.r_d > sl.r_f  # USD over EUR
    assert all(sl.convention in ("spot", "forward") for sl in m.slices)


def test_usdjpy_preset_shape():
    m = usdjpy_market()
    for sl in m.slices:
        assert sl.quotes.rr25 < -0.005  # big negative RR
        assert sl.convention.endswith("_pa")  # premium-adjusted quotes
        assert sl.r_f > sl.r_d  # USD rate over JPY rate
    # skew grows with tenor
    rrs = [sl.quotes.rr25 for sl in m.slices]
    assert rrs[0] > rrs[-1]


def test_em_preset_shape():
    m = em_high_vol_market()
    for sl in m.slices:
        assert sl.quotes.atm >= 0.33  # extreme vol regime
        assert sl.quotes.rr25 > 0.0  # devaluation skew: calls rich
        assert sl.r_d > 0.25  # EM carry


def test_ground_truth_quotes_consistent_with_heston():
    """The generated quotes, re-expanded to pillar vols and strikes, must
    reproduce the Heston implied vols at those strikes."""
    params = HestonParams(v0=0.0064, kappa=1.8, theta=0.008, xi=0.45, rho=-0.35)
    m = market_from_heston(params, tenors=(("3m", 91 / 365), ("1y", 1.0)))
    for cal in calibration_slices(m):
        for K, v in zip(cal.strikes, cal.vols):
            pr = float(price_cos(m.S, float(K), cal.T, cal.r_d, cal.r_f, params, -1))
            iv = implied_vol(pr, m.S, float(K), cal.T, cal.r_d, cal.r_f, -1)
            assert iv == pytest.approx(v, abs=5e-7)


def test_ground_truth_mode_seeded_noise():
    params = HestonParams(v0=0.0064, kappa=1.8, theta=0.008, xi=0.45, rho=-0.35)
    a = market_from_heston(params, tenors=(("1m", 30 / 365),), noise=0.001, seed=9)
    b = market_from_heston(params, tenors=(("1m", 30 / 365),), noise=0.001, seed=9)
    assert a.slices[0].quotes == b.slices[0].quotes
    clean = market_from_heston(params, tenors=(("1m", 30 / 365),))
    assert a.slices[0].quotes != clean.slices[0].quotes


def test_calibration_slices_structure(eurusd):
    slices = calibration_slices(eurusd)
    assert len(slices) == 6
    for cal, ms in zip(slices, eurusd.slices):
        assert cal.T == ms.T
        assert cal.strikes.shape == cal.vols.shape == (5,)
        assert np.all(np.diff(cal.strikes) > 0)
