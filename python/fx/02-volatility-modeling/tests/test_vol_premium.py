"""Vol risk premium calculator: forward realized vol, premium, var-swap P&L."""

import numpy as np
import pytest

from fx_vol import (
    premium_summary,
    realized_vol_forward,
    variance_swap_pnl,
    vol_risk_premium,
)
from fx_vol.data import synthetic as syn


class TestRealizedForward:
    def test_manual_window(self):
        r = np.array([0.01, -0.02, 0.015, -0.005, 0.01])
        rv = realized_vol_forward(r, window=2, periods_per_year=252)
        # rv[0] covers r[1], r[2]
        expected0 = np.sqrt(252 * np.mean([r[1] ** 2, r[2] ** 2]))
        assert rv[0] == pytest.approx(expected0, rel=1e-12)
        # incomplete windows are NaN
        assert np.isnan(rv[-1]) and np.isnan(rv[-2])

    def test_recovers_constant_vol(self):
        vol = 0.006
        r = syn.simulate_constant_vol(20_000, vol, seed=101)
        rv = realized_vol_forward(r, window=21)
        valid = rv[np.isfinite(rv)]
        assert np.mean(valid) == pytest.approx(vol * np.sqrt(252), rel=0.02)

    def test_validation(self):
        r = syn.simulate_constant_vol(100, 0.006, seed=102)
        with pytest.raises(ValueError, match="window"):
            realized_vol_forward(r, window=1)
        with pytest.raises(ValueError, match="window"):
            realized_vol_forward(r, window=101)


class TestPremium:
    def test_recovers_injected_spread(self):
        """Synthetic implied = subsequent realized + 1.5 vol points: the
        estimated premium must average ~0.015."""
        r = syn.simulate_constant_vol(10_000, 0.006, seed=103)
        rv = realized_vol_forward(r, window=21)
        iv = np.where(np.isfinite(rv), rv, 0.10) + 0.015
        prem = vol_risk_premium(iv, rv)
        summary = premium_summary(prem)
        assert summary["mean"] == pytest.approx(0.015, abs=1e-6)
        assert summary["frac_positive"] == 1.0

    def test_nan_propagation(self):
        prem = vol_risk_premium(np.array([0.10, 0.11]), np.array([0.08, np.nan]))
        assert prem[0] == pytest.approx(0.02)
        assert np.isnan(prem[1])

    def test_alignment_enforced(self):
        with pytest.raises(ValueError, match="aligned"):
            vol_risk_premium([0.1, 0.1], [0.1])
        with pytest.raises(ValueError, match="non-negative"):
            vol_risk_premium([-0.1, 0.1], [0.1, 0.1])


class TestVarianceSwapPnl:
    def test_sign_convention(self):
        iv = np.array([0.10, 0.10, 0.10])
        rv = np.array([0.08, 0.10, 0.15])
        pnl = variance_swap_pnl(iv, rv, vega_notional=1.0)
        assert pnl[0] > 0          # realized below strike: seller profits
        assert pnl[1] == pytest.approx(0.0, abs=1e-15)
        assert pnl[2] < 0          # crisis: seller loses
        # convexity works against the seller: symmetric vol moves are not symmetric in P&L
        assert abs(pnl[2]) > abs(variance_swap_pnl([0.10], [0.05], 1.0)[0])

    def test_vega_units_at_strike(self):
        """Near the strike, 1 vol point of realized ~ 1 unit of vega notional."""
        pnl_lo = variance_swap_pnl([0.10], [0.095], vega_notional=1.0)[0]
        pnl_hi = variance_swap_pnl([0.10], [0.105], vega_notional=1.0)[0]
        # d pnl / d vol_point ~ -1 * (rv/iv) ~ -1 near ATM strike
        assert (pnl_lo - pnl_hi) / 0.01 == pytest.approx(1.0, rel=0.06)

    def test_validation(self):
        with pytest.raises(ValueError, match="positive"):
            variance_swap_pnl([0.0], [0.1])


class TestSummary:
    def test_summary_statistics(self):
        prem = np.array([0.01, 0.02, -0.01, np.nan, 0.03])
        s = premium_summary(prem)
        assert s["n"] == 4
        assert s["mean"] == pytest.approx(0.0125)
        assert s["frac_positive"] == pytest.approx(0.75)
        assert s["min"] == pytest.approx(-0.01)

    def test_all_nan_rejected(self):
        with pytest.raises(ValueError, match="no finite"):
            premium_summary([np.nan, np.nan])
