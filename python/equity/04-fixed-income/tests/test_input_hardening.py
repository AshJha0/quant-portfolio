"""Non-finite market inputs must raise ``ValueError`` -- never a NaN price.

Every comparison with NaN is ``False``, so ``if 1 + ytm/m <= 0: raise``
silently accepted a NaN yield and ``price_from_ytm`` returned ``nan``;
``FixedRateBond(coupon=nan)`` was accepted and ``dv01`` came back ``nan``;
``ytm_from_price(nan)`` failed only deep inside ``brentq`` with SciPy's
message.  Every public scalar market input now goes through
``_require_finite`` first, so the error names the offending argument.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

import fi_rates as fr

NON_FINITE = [math.nan, math.inf, -math.inf]
D = dt.date


@pytest.mark.parametrize("bad", NON_FINITE)
class TestBondConstruction:
    def test_fixed_rate_bond_rejects_non_finite_coupon(self, settlement, bad):
        with pytest.raises(ValueError, match="coupon must be finite"):
            fr.FixedRateBond(effective=settlement, maturity=D(2031, 8, 18),
                             coupon=bad, frequency=2)

    def test_fixed_rate_bond_rejects_non_finite_face(self, settlement, bad):
        with pytest.raises(ValueError, match="face must be finite"):
            fr.FixedRateBond(effective=settlement, maturity=D(2031, 8, 18),
                             coupon=0.04, frequency=2, face=bad)

    def test_zero_coupon_bond_rejects_non_finite_face(self, bad):
        with pytest.raises(ValueError, match="face must be finite"):
            fr.ZeroCouponBond(maturity=D(2031, 8, 18), face=bad)


@pytest.mark.parametrize("bad", NON_FINITE)
class TestYieldAnalytics:
    def test_price_from_ytm(self, settlement, bond_5y, bad):
        with pytest.raises(ValueError, match="ytm must be finite"):
            fr.price_from_ytm(bond_5y, settlement, bad)

    def test_ytm_from_price(self, settlement, bond_5y, bad):
        with pytest.raises(ValueError, match="clean_price must be finite"):
            fr.ytm_from_price(bond_5y, settlement, bad)

    def test_duration_convexity_dv01(self, settlement, bond_5y, bad):
        for fn in (fr.macaulay_duration, fr.modified_duration, fr.convexity, fr.dv01):
            with pytest.raises(ValueError, match="ytm must be finite"):
                fn(bond_5y, settlement, bad)

    def test_z_spread_inputs(self, settlement, bond_5y, curve, bad):
        with pytest.raises(ValueError, match="z_spread must be finite"):
            fr.dirty_price_from_curve(bond_5y, settlement, curve, z_spread=bad)
        with pytest.raises(ValueError, match="clean_price must be finite"):
            fr.z_spread_from_price(bond_5y, settlement, curve, bad)

    def test_frn_and_annuity_inputs(self, settlement, flat_curve, bad):
        with pytest.raises(ValueError, match="quoted_margin must be finite"):
            fr.frn_price_from_curve(settlement, D(2031, 8, 18), flat_curve,
                                    quoted_margin=bad)
        with pytest.raises(ValueError, match="face must be finite"):
            fr.frn_price_from_curve(settlement, D(2031, 8, 18), flat_curve, face=bad)
        with pytest.raises(ValueError, match="payment must be finite"):
            fr.annuity_pv(bad, settlement, D(2031, 8, 18), flat_curve)


class TestGuardsStillCatchTheOldCases:
    def test_yield_below_minus_frequency_still_rejected(self, settlement, bond_5y):
        with pytest.raises(ValueError, match="1 \\+ ytm/frequency"):
            fr.price_from_ytm(bond_5y, settlement, -2.5)
        with pytest.raises(ValueError, match="1 \\+ ytm/frequency"):
            fr.macaulay_duration(bond_5y, settlement, -2.5)
        with pytest.raises(ValueError, match="1 \\+ ytm/frequency"):
            fr.convexity(bond_5y, settlement, -2.5)

    def test_frn_non_positive_face_rejected(self, settlement, flat_curve):
        with pytest.raises(ValueError, match="face must be > 0"):
            fr.frn_price_from_curve(settlement, D(2031, 8, 18), flat_curve, face=0.0)

    def test_clean_inputs_unchanged(self, settlement, bond_5y):
        # Par bond at its own coupon on a coupon date prices to par.
        assert fr.price_from_ytm(bond_5y, settlement, 0.04) == pytest.approx(100.0, abs=1e-10)
