"""FX delta conventions, conversions, strike-from-delta, ATM solvers."""

import math

import pytest

from fx_options import (CONVENTIONS, atm_dns_strike, atm_forward_strike,
                        delta, forward_to_spot_delta, gk_price,
                        premium_adjust_spot_delta, spot_to_forward_delta,
                        strike_from_delta)

MKT = dict(S=1.10, T=0.5, r_d=0.0425, r_f=0.0290, sigma=0.0825)
JPY = dict(S=147.5, T=0.5, r_d=0.0050, r_f=0.0525, sigma=0.1075)


def _delta(mkt, K, ot, conv):
    return delta(mkt["S"], K, mkt["T"], mkt["r_d"], mkt["r_f"],
                 mkt["sigma"], ot, conv)


class TestConventionRelations:
    @pytest.mark.parametrize("mkt", [MKT, JPY])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_forward_equals_spot_times_erf(self, mkt, option_type):
        K = mkt["S"] * 1.02
        ds = _delta(mkt, K, option_type, "spot")
        df = _delta(mkt, K, option_type, "forward")
        assert df == pytest.approx(ds * math.exp(mkt["r_f"] * mkt["T"]),
                                   abs=1e-14)
        assert spot_to_forward_delta(ds, mkt["T"], mkt["r_f"]) == pytest.approx(df)
        assert forward_to_spot_delta(df, mkt["T"], mkt["r_f"]) == pytest.approx(ds)

    @pytest.mark.parametrize("mkt", [MKT, JPY])
    def test_premium_adjusted_below_unadjusted_for_calls(self, mkt):
        for k_mult in (0.9, 1.0, 1.1):
            K = mkt["S"] * k_mult
            assert _delta(mkt, K, "call", "spot_pa") < _delta(mkt, K, "call", "spot")
            assert _delta(mkt, K, "call", "forward_pa") < _delta(mkt, K, "call", "forward")

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_premium_adjustment_identity(self, option_type):
        # delta_pa_spot = delta_spot - V/S exactly.
        K = MKT["S"] * 0.97
        price = gk_price(MKT["S"], K, MKT["T"], MKT["r_d"], MKT["r_f"],
                         MKT["sigma"], option_type)
        ds = _delta(MKT, K, option_type, "spot")
        dpa = _delta(MKT, K, option_type, "spot_pa")
        assert dpa == pytest.approx(
            premium_adjust_spot_delta(ds, price, MKT["S"]), abs=1e-14)

    def test_put_deltas_negative_call_positive(self):
        K = MKT["S"]
        for conv in CONVENTIONS:
            assert _delta(MKT, K, "call", conv) > 0
            assert _delta(MKT, K, "put", conv) < 0

    def test_delta_matches_finite_difference(self):
        # Spot delta is dV/dS.
        K, h = 1.12, 1e-6
        up = gk_price(MKT["S"] + h, K, MKT["T"], MKT["r_d"], MKT["r_f"],
                      MKT["sigma"], "call")
        dn = gk_price(MKT["S"] - h, K, MKT["T"], MKT["r_d"], MKT["r_f"],
                      MKT["sigma"], "call")
        assert _delta(MKT, K, "call", "spot") == pytest.approx(
            (up - dn) / (2 * h), abs=1e-7)


class TestStrikeFromDelta:
    @pytest.mark.parametrize("mkt", [MKT, JPY])
    @pytest.mark.parametrize("conv", CONVENTIONS)
    @pytest.mark.parametrize("target,ot", [
        (0.25, "call"), (0.10, "call"), (0.45, "call"),
        (-0.25, "put"), (-0.10, "put"), (-0.45, "put"),
    ])
    def test_round_trip_all_conventions(self, mkt, conv, target, ot):
        K = strike_from_delta(target, mkt["S"], mkt["T"], mkt["r_d"],
                              mkt["r_f"], mkt["sigma"], ot, conv)
        assert _delta(mkt, K, ot, conv) == pytest.approx(target, abs=1e-8)

    def test_pa_call_takes_larger_strike_branch(self):
        # The 25d PA call strike must be OTM (above the ATM-DNS-pa strike),
        # i.e. on the decreasing branch of K -> (K/F) N(d2).
        K = strike_from_delta(0.25, MKT["S"], MKT["T"], MKT["r_d"],
                              MKT["r_f"], MKT["sigma"], "call", "forward_pa")
        k_dns_pa = atm_dns_strike(**MKT, convention="forward_pa")
        assert K > k_dns_pa

    def test_pa_call_delta_above_max_raises(self):
        with pytest.raises(ValueError, match="maximum attainable"):
            strike_from_delta(0.99, MKT["S"], MKT["T"], MKT["r_d"],
                              MKT["r_f"], MKT["sigma"], "call", "forward_pa")

    def test_wrong_sign_raises(self):
        with pytest.raises(ValueError, match="sign"):
            strike_from_delta(-0.25, MKT["S"], MKT["T"], MKT["r_d"],
                              MKT["r_f"], MKT["sigma"], "call", "spot")
        with pytest.raises(ValueError, match="sign"):
            strike_from_delta(0.25, MKT["S"], MKT["T"], MKT["r_d"],
                              MKT["r_f"], MKT["sigma"], "put", "spot")

    def test_out_of_range_delta_raises(self):
        with pytest.raises(ValueError):
            strike_from_delta(1.5, MKT["S"], MKT["T"], MKT["r_d"],
                              MKT["r_f"], MKT["sigma"], "call", "forward")

    def test_unknown_convention_raises(self):
        with pytest.raises(ValueError, match="convention"):
            strike_from_delta(0.25, MKT["S"], MKT["T"], MKT["r_d"],
                              MKT["r_f"], MKT["sigma"], "call", "vega")

    def test_t_zero_raises(self):
        with pytest.raises(ValueError):
            strike_from_delta(0.25, 1.1, 0.0, 0.03, 0.01, 0.1, "call", "spot")


class TestATMConventions:
    def test_atm_forward_strike_is_cip_forward(self):
        K = atm_forward_strike(MKT["S"], MKT["T"], MKT["r_d"], MKT["r_f"])
        assert K == pytest.approx(
            MKT["S"] * math.exp((MKT["r_d"] - MKT["r_f"]) * MKT["T"]))

    @pytest.mark.parametrize("conv", CONVENTIONS)
    @pytest.mark.parametrize("mkt", [MKT, JPY])
    def test_dns_strike_zeroes_straddle_delta(self, conv, mkt):
        K = atm_dns_strike(mkt["S"], mkt["T"], mkt["r_d"], mkt["r_f"],
                           mkt["sigma"], conv)
        straddle = (_delta(mkt, K, "call", conv) + _delta(mkt, K, "put", conv))
        assert abs(straddle) < 1e-12

    def test_dns_formulae(self):
        F = atm_forward_strike(MKT["S"], MKT["T"], MKT["r_d"], MKT["r_f"])
        w = 0.5 * MKT["sigma"] ** 2 * MKT["T"]
        assert atm_dns_strike(**MKT, convention="spot") == pytest.approx(
            F * math.exp(w), abs=1e-14)
        assert atm_dns_strike(**MKT, convention="forward_pa") == pytest.approx(
            F * math.exp(-w), abs=1e-14)

    def test_dns_above_forward_unadjusted_below_for_pa(self):
        F = atm_forward_strike(MKT["S"], MKT["T"], MKT["r_d"], MKT["r_f"])
        assert atm_dns_strike(**MKT, convention="forward") > F
        assert atm_dns_strike(**MKT, convention="spot_pa") < F


class TestPremiumAdjustedCallFold:
    """TEST-6: the premium-adjusted call delta ``(K/F) N(d2)`` is not
    injective in ``K`` -- it rises to a peak at ``_pa_peak_strike`` and then
    decays.  ``strike_from_delta`` must (a) refuse a delta above the peak,
    (b) return the *larger-strike* root when two strikes share a delta, and
    (c) reproduce the target to solver precision.
    """

    @pytest.mark.parametrize("mkt", [MKT, JPY])
    @pytest.mark.parametrize("conv", ["spot_pa", "forward_pa"])
    def test_peak_is_a_true_maximum_of_the_pa_delta(self, mkt, conv):
        from fx_options.deltas import _pa_peak_strike

        F = mkt["S"] * math.exp((mkt["r_d"] - mkt["r_f"]) * mkt["T"])
        k_peak = _pa_peak_strike(F, mkt["T"], mkt["sigma"])
        d_peak = _delta(mkt, k_peak, "call", conv)
        for k in (k_peak * 0.9, k_peak * 0.99, k_peak * 1.01, k_peak * 1.1):
            assert _delta(mkt, k, "call", conv) < d_peak

    @pytest.mark.parametrize("mkt", [MKT, JPY])
    @pytest.mark.parametrize("conv", ["spot_pa", "forward_pa"])
    def test_delta_just_above_peak_raises_just_below_solves(self, mkt, conv):
        from fx_options.deltas import _pa_peak_strike

        F = mkt["S"] * math.exp((mkt["r_d"] - mkt["r_f"]) * mkt["T"])
        k_peak = _pa_peak_strike(F, mkt["T"], mkt["sigma"])
        d_max = _delta(mkt, k_peak, "call", conv)
        with pytest.raises(ValueError, match="maximum attainable"):
            strike_from_delta(d_max * (1 + 1e-6), mkt["S"], mkt["T"], mkt["r_d"],
                              mkt["r_f"], mkt["sigma"], "call", conv)
        K = strike_from_delta(d_max * (1 - 1e-4), mkt["S"], mkt["T"], mkt["r_d"],
                              mkt["r_f"], mkt["sigma"], "call", conv)
        assert K >= k_peak
        assert _delta(mkt, K, "call", conv) == pytest.approx(d_max * (1 - 1e-4), abs=1e-12)

    @pytest.mark.parametrize("mkt", [MKT, JPY])
    @pytest.mark.parametrize("conv", ["spot_pa", "forward_pa"])
    @pytest.mark.parametrize("frac", [0.3, 0.6, 0.9])
    def test_twice_attainable_delta_returns_larger_strike_root(self, mkt, conv, frac):
        from scipy.optimize import brentq

        from fx_options.deltas import _pa_peak_strike

        F = mkt["S"] * math.exp((mkt["r_d"] - mkt["r_f"]) * mkt["T"])
        k_peak = _pa_peak_strike(F, mkt["T"], mkt["sigma"])
        d_max = _delta(mkt, k_peak, "call", conv)
        target = frac * d_max
        K = strike_from_delta(target, mkt["S"], mkt["T"], mkt["r_d"],
                              mkt["r_f"], mkt["sigma"], "call", conv)
        # (c) exact round trip on the returned strike ...
        assert _delta(mkt, K, "call", conv) == pytest.approx(target, abs=1e-12)
        # (b) ... which lies on the decreasing branch ...
        assert K >= k_peak
        # ... while an independent root on the *left* branch also exists and
        # is not the one returned.
        v = mkt["sigma"] * math.sqrt(mkt["T"])
        k_left = brentq(lambda k: _delta(mkt, k, "call", conv) - target,
                        F * math.exp(-30 * v), k_peak, xtol=1e-14)
        assert k_left < k_peak < K
        assert _delta(mkt, k_left, "call", conv) == pytest.approx(target, abs=1e-10)
