"""Key-rate durations: partition of unity, sum vs parallel, locality."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import fi_rates as fr
from fi_rates.keyrates import DEFAULT_KEY_TENORS, triangle_weights

D = dt.date


class TestTriangleWeights:
    def test_partition_of_unity(self, curve):
        w = triangle_weights(DEFAULT_KEY_TENORS, curve.times)
        np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-14)

    def test_peak_is_one_at_key_tenor(self):
        w = triangle_weights((2.0, 5.0, 10.0), np.array([2.0, 5.0, 10.0]))
        np.testing.assert_allclose(np.diag(w), 1.0, atol=1e-15)

    def test_flat_extension_at_ends(self):
        w = triangle_weights((2.0, 5.0), np.array([0.5, 30.0]))
        assert w[0, 0] == 1.0  # short times fully on first key
        assert w[1, 1] == 1.0  # long times fully on last key

    def test_non_increasing_tenors_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            triangle_weights((5.0, 2.0), np.array([1.0]))


class TestKRDSum:
    def test_krd_sum_matches_parallel_dv01_bond(self, settlement, curve, bond_5y):
        krdv = fr.key_rate_dv01s(bond_5y, settlement, curve)
        parallel = fr.dv01_curve(bond_5y, settlement, curve)
        # partition of unity => equality up to cross-gamma of the 1bp bumps
        assert krdv.sum() == pytest.approx(parallel, rel=1e-6)

    def test_krd_sum_matches_parallel_dv01_portfolio(
        self, settlement, curve, portfolio
    ):
        krdv = fr.key_rate_dv01s(portfolio, settlement, curve)
        rpt = fr.portfolio_risk(portfolio, settlement, curve)
        parallel = float(rpt.loc["TOTAL", "dv01"])
        assert krdv.sum() == pytest.approx(parallel, rel=1e-6)

    def test_krd_sum_documented_tolerance_under_pchip(self, settlement, quotes):
        """Non-local interpolation: match holds only to a looser tolerance."""
        curve = fr.bootstrap_curve(quotes, interpolation="pchip_zero")
        bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2036, 8, 18), coupon=0.04, frequency=2
        )
        krdv = fr.key_rate_dv01s(bond, settlement, curve)
        parallel = fr.dv01_curve(bond, settlement, curve)
        assert krdv.sum() == pytest.approx(parallel, rel=1e-4)

    def test_krd_report_sum_row(self, settlement, curve, portfolio):
        rpt = fr.krd_report(portfolio, settlement, curve)
        body = rpt.drop(index="SUM")
        assert rpt.loc["SUM", "key_rate_dv01"] == pytest.approx(
            float(body["key_rate_dv01"].sum()), abs=1e-12
        )


class TestLocality:
    def test_far_pillar_bump_barely_affects_short_bond(self, settlement, curve):
        short_bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2028, 8, 18), coupon=0.03, frequency=2
        )
        krdv = fr.key_rate_dv01s(short_bond, settlement, curve)
        # 2y bond: essentially all risk at the 2y key rate; none at 10y/30y
        assert krdv[0] > 100 * abs(krdv[2])
        assert abs(krdv[3]) < 1e-6 * krdv[0] + 1e-9

    def test_zcb_krd_concentrated_at_maturity_pillar(self, settlement, curve):
        zcb = fr.FixedRateBond(
            effective=settlement,
            maturity=D(2031, 8, 18),
            coupon=0.0,
            frequency=1,
        )
        krd = fr.key_rate_durations(zcb, settlement, curve)  # keys 2/5/10/30
        # 5y ZCB: everything at the 5y key rate
        assert krd[1] == pytest.approx(krd.sum(), rel=1e-3)
        assert krd[1] > 4.5
        assert abs(krd[3]) < 1e-6

    def test_long_bond_loads_long_keys(self, settlement, curve):
        long_bond = fr.FixedRateBond(
            effective=settlement, maturity=D(2056, 8, 18), coupon=0.045, frequency=2
        )
        krdv = fr.key_rate_dv01s(long_bond, settlement, curve)
        assert krdv[3] > krdv[0]  # 30y key dominates 2y key


class TestKeyRateConvexity:
    def test_zcb_convexity_positive_at_its_pillar(self, settlement, curve):
        zcb = fr.FixedRateBond(
            effective=settlement, maturity=D(2031, 8, 18), coupon=0.0, frequency=1
        )
        krc = fr.key_rate_convexities(zcb, settlement, curve)
        assert krc[1] > 0.0
        assert krc[1] > abs(krc[3])

    def test_custom_key_tenor_set(self, settlement, curve, bond_5y):
        keys = (1.0, 3.0, 7.0, 20.0)
        krdv = fr.key_rate_dv01s(bond_5y, settlement, curve, key_tenors=keys)
        parallel = fr.dv01_curve(bond_5y, settlement, curve)
        assert krdv.sum() == pytest.approx(parallel, rel=1e-6)
