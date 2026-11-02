"""Funded carry, pull-to-par split, the GFC 2008 shock and PCHIP smoothness."""

import datetime as dt

import fi_rates as fr
import numpy as np
import pytest

D = dt.date


class TestFundedCarry:
    def test_funding_is_dirty_price_times_repo_act360(self, settlement, curve, bond_5y):
        horizon = D(2027, 2, 18)
        base = fr.carry_rolldown(bond_5y, settlement, horizon, curve)
        funded = fr.carry_rolldown(bond_5y, settlement, horizon, curve, repo_rate=0.05)
        days = (horizon - settlement).days
        assert funded["funding"] == pytest.approx(base["price_start"] * 0.05 * days / 360,
                                                  rel=1e-14)
        assert funded["carry"] == pytest.approx(base["carry"] - funded["funding"], rel=1e-14)
        assert funded["total"] == pytest.approx(funded["carry"] + funded["rolldown"], rel=1e-14)

    def test_repo_above_coupon_gives_negative_carry(self, settlement, curve, bond_5y):
        out = fr.carry_rolldown(bond_5y, settlement, D(2027, 8, 18), curve, repo_rate=0.08)
        assert out["coupon_income"] > 0 > out["carry"]

    def test_flat_curve_rolldown_is_all_pull_to_par(self, settlement, flat_curve):
        # on a flat continuous curve the bond's own yield is the curve rate,
        # so the static clean-price change is pure pull-to-par
        bond = fr.FixedRateBond(settlement, D(2031, 8, 18), 0.06, 2)
        out = fr.carry_rolldown(bond, settlement, D(2028, 8, 18), flat_curve)
        assert out["curve_rolldown"] == pytest.approx(0.0, abs=1e-9)
        assert out["pull_to_par"] < 0  # premium bond pulls down to par

    def test_upward_curve_has_positive_curve_rolldown(self, settlement, curve, bond_5y):
        out = fr.carry_rolldown(bond_5y, settlement, D(2027, 8, 18), curve)
        assert out["curve_rolldown"] > 0

    def test_non_finite_repo_rejected(self, settlement, curve, bond_5y):
        with pytest.raises(ValueError, match="repo_rate"):
            fr.carry_rolldown(bond_5y, settlement, D(2027, 8, 18), curve, repo_rate=np.nan)


def test_gfc_2008_long_end_rallied_with_the_10y():
    """Year-end 2007 to year-end 2008 UST CMT: 10y -179bp, 30y -176bp.

    The old encoding had 30y -75bp, understating long-end P&L by ~100bp.
    """
    sc = fr.HISTORICAL_SCENARIOS["gfc_2008"]
    shifts = dict(zip(sc.tenors, sc.shifts_bp, strict=False))
    assert abs(shifts[30.0] - shifts[10.0]) <= 10.0
    assert shifts[30.0] <= -150.0


def test_pchip_forward_is_continuous_but_kinked_at_pillars():
    """PCHIP on zeros gives C1 zeros but only C0 forwards f = z + t z'."""
    times = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
    c = fr.DiscountCurve.from_zero_rates(
        times, [0.02, 0.024, 0.028, 0.031, 0.034, 0.036, 0.037], "pchip_zero"
    )

    def inst_fwd(t, h=1e-6):
        return c.forward_rate(t - h, t + h)

    eps = 1e-4
    left, right = inst_fwd(5.0 - eps), inst_fwd(5.0 + eps)
    assert abs(right - left) < 1e-4                    # continuous
    slope_l = (inst_fwd(5.0 - eps) - inst_fwd(5.0 - 2 * eps)) / eps
    slope_r = (inst_fwd(5.0 + 2 * eps) - inst_fwd(5.0 + eps)) / eps
    assert abs(slope_r - slope_l) > 1e-4              # derivative jumps: not C1


class TestFrnCurrentFixing:
    def test_mid_period_frn_identity(self, settlement, curve):
        """Zero margin: dirty = P(t1) * face * (1 + fixing * ACT/360)."""
        mat = D(2031, 10, 1)
        price = fr.frn_price_from_curve(settlement, mat, curve, 4, current_fixing=0.05)
        t1 = D(2026, 10, 1)
        p1 = float(curve.df(fr.curve_time(settlement, t1)))
        tau = (t1 - D(2026, 7, 1)).days / 360.0
        assert price == pytest.approx(p1 * 100 * (1 + 0.05 * tau), rel=1e-12)

    def test_without_fixing_unchanged(self, settlement, curve):
        mat = D(2031, 8, 18)
        assert fr.frn_price_from_curve(settlement, mat, curve, 4) == pytest.approx(100.0,
                                                                                   abs=1e-10)

    def test_non_finite_fixing_rejected(self, settlement, curve):
        with pytest.raises(ValueError, match="current_fixing"):
            fr.frn_price_from_curve(settlement, D(2031, 10, 1), curve, 4,
                                    current_fixing=float("nan"))
