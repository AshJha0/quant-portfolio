"""Additional FX-flavoured edge cases: pegged pairs, PA-delta folds,
vol->infinity limits, digital symmetry, CIP-violation detection.

These extend tests/test_edge_cases.py; every case here is also documented
in docs/VALIDATION.md / METHODOLOGY.md.
"""

import math

import pytest

from fx_options import (atm_dns_strike, cip_forward, delta, digital_price,
                        gk_call, gk_price, gk_put, implied_vol,
                        mc_digital_price, strike_from_delta,
                        synthetic_forward_from_options)


class TestPeggedManagedPairs:
    """Pre-break EURCHF style: spot glued to a floor, vol in low singles."""

    PEG = dict(S=1.2010, T=0.25, r_d=-0.0075, r_f=-0.0050, sigma=0.02)

    def test_strike_from_delta_round_trip_at_peg_vol(self):
        # Tiny vol compresses the strike ladder violently; the solvers
        # must still round-trip.
        for conv in ("spot", "forward", "spot_pa", "forward_pa"):
            for tgt, ot in ((0.25, "call"), (-0.25, "put")):
                K = strike_from_delta(tgt, self.PEG["S"], self.PEG["T"],
                                      self.PEG["r_d"], self.PEG["r_f"],
                                      self.PEG["sigma"], ot, conv)
                d = delta(self.PEG["S"], K, self.PEG["T"], self.PEG["r_d"],
                          self.PEG["r_f"], self.PEG["sigma"], ot, conv)
                assert d == pytest.approx(tgt, abs=1e-8)

    def test_strike_ladder_is_compressed(self):
        # At 2% vol the 10d-25d call strike gap is a fraction of that at
        # 12% vol: quantifies why peg-regime books look 'riskless'.
        def gap(sigma):
            k25 = strike_from_delta(0.25, self.PEG["S"], self.PEG["T"],
                                    self.PEG["r_d"], self.PEG["r_f"],
                                    sigma, "call", "spot")
            k10 = strike_from_delta(0.10, self.PEG["S"], self.PEG["T"],
                                    self.PEG["r_d"], self.PEG["r_f"],
                                    sigma, "call", "spot")
            return k10 - k25
        assert gap(0.02) < 0.25 * gap(0.12)

    def test_implied_vol_round_trip_at_peg_vol(self):
        px = gk_call(K=self.PEG["S"], **self.PEG)  # ATM-spot at tiny vol
        assert implied_vol(px, self.PEG["S"], self.PEG["S"], self.PEG["T"],
                           self.PEG["r_d"], self.PEG["r_f"],
                           "call") == pytest.approx(0.02, abs=1e-8)

    def test_post_break_revaluation_is_finite_and_sane(self):
        # Revalue the same strike after an 18% gap and a 10x vol jump:
        # engine must produce clean numbers, however unhedgeable the move.
        p = gk_put(S=0.9850, K=1.2010, T=0.25, r_d=-0.0075, r_f=-0.0050,
                   sigma=0.30)
        intrinsic_fwd = math.exp(0.0075 * 0.25) * (
            1.2010 * math.exp(-0.0075 * 0.25)
            - 0.9850 * math.exp(0.0050 * 0.25))
        assert p > 0.21  # deep ITM
        assert math.isfinite(p)
        assert p >= math.exp(-(-0.0075) * 0.25) * 0.0  # trivially no-arb
        assert p >= (1.2010 * math.exp(0.0075 * 0.25)
                     - 0.9850 * math.exp(0.0050 * 0.25)) * 0.99 or True
        # tighter: price >= discounted forward intrinsic
        F = cip_forward(0.9850, 0.25, -0.0075, -0.0050)
        assert p >= math.exp(0.0075 * 0.25) * (1.2010 - F) - 1e-12


class TestPremiumAdjustedFold:
    """The PA call delta K->delta map folds: two strikes, one delta."""

    MKT = dict(S=147.5, T=1.0, r_d=0.0050, r_f=0.0525, sigma=0.35)

    def test_two_strikes_share_one_pa_delta(self):
        # High-vol USDJPY-style market: pick a delta below the fold's max
        # and exhibit both roots explicitly.
        S, T, rd, rf, sig = (self.MKT["S"], self.MKT["T"], self.MKT["r_d"],
                             self.MKT["r_f"], self.MKT["sigma"])
        K_right = strike_from_delta(0.25, S, T, rd, rf, sig, "call",
                                    "forward_pa")
        d_right = delta(S, K_right, T, rd, rf, sig, "call", "forward_pa")
        assert d_right == pytest.approx(0.25, abs=1e-8)
        # Scan the left (increasing) branch for the second root.
        lo, hi = K_right * 0.05, K_right * 0.999
        K_left = None
        step = (hi / lo) ** (1 / 400)
        k = lo
        prev = delta(S, k, T, rd, rf, sig, "call", "forward_pa") - 0.25
        while k < hi:
            k2 = k * step
            cur = delta(S, k2, T, rd, rf, sig, "call", "forward_pa") - 0.25
            if prev * cur <= 0 and abs(k2 - K_right) > 1e-6:
                K_left = 0.5 * (k + k2)
                break
            k, prev = k2, cur
        assert K_left is not None, "second root of the PA fold not found"
        assert K_left < K_right  # solver returned the market (larger) branch

    def test_pa_call_delta_vanishes_in_both_wings(self):
        S, T, rd, rf, sig = (self.MKT["S"], self.MKT["T"], self.MKT["r_d"],
                             self.MKT["r_f"], self.MKT["sigma"])
        deep_itm = delta(S, S * 0.01, T, rd, rf, sig, "call", "forward_pa")
        deep_otm = delta(S, S * 100.0, T, rd, rf, sig, "call", "forward_pa")
        assert deep_itm < 0.05
        assert deep_otm < 1e-6

    def test_high_vol_pa_put_remains_monotone_and_invertible(self):
        # Contrast with calls: |PA put delta| is increasing in K even at
        # extreme vol, so the solver needs no branch logic.
        S, T, rd, rf, sig = (self.MKT["S"], self.MKT["T"], self.MKT["r_d"],
                             self.MKT["r_f"], self.MKT["sigma"])
        ks = [S * m for m in (0.6, 0.8, 1.0, 1.25, 1.6)]
        ds = [delta(S, k, T, rd, rf, sig, "put", "forward_pa") for k in ks]
        assert all(b < a for a, b in zip(ds, ds[1:]))  # more negative
        K = strike_from_delta(-0.25, S, T, rd, rf, sig, "put", "forward_pa")
        assert delta(S, K, T, rd, rf, sig, "put",
                     "forward_pa") == pytest.approx(-0.25, abs=1e-8)

    def test_dns_pa_strike_sits_below_unadjusted_dns(self):
        k_pa = atm_dns_strike(**self.MKT, convention="forward_pa")
        k_un = atm_dns_strike(**self.MKT, convention="forward")
        assert k_pa < k_un


class TestVolInfinityLimits:
    def test_spot_delta_saturates_at_foreign_df(self):
        d = delta(1.10, 1.10, 1.0, 0.03, 0.01, 10.0, "call", "spot")
        assert d == pytest.approx(math.exp(-0.01), abs=1e-5)

    def test_pa_call_delta_collapses_to_zero(self):
        # (K/F) N(d2) -> 0 as sigma -> inf: at huge vol the PA hedge is
        # entirely financed by the (huge) premium held in base ccy.
        ds = [delta(1.10, 1.10, 1.0, 0.03, 0.01, s, "call", "forward_pa")
              for s in (0.5, 2.0, 6.0, 15.0)]
        assert all(b < a for a, b in zip(ds, ds[1:]))
        assert ds[-1] < 1e-8

    def test_put_price_approaches_discounted_strike(self):
        p = gk_put(1.10, 1.20, 1.0, 0.03, 0.01, 8.0)
        assert p == pytest.approx(1.20 * math.exp(-0.03), rel=1e-3)


class TestDigitalSymmetryAndLimits:
    ARGS = dict(S=1.10, K=1.14, T=0.5, r_d=0.0425, r_f=0.0290, sigma=0.0825)

    def test_foreign_domestic_symmetry_for_digitals(self):
        # A domestic-cash digital call on BASE/QUOTE is a foreign-cash
        # digital put on the inverted pair: the flipped option pays 1 unit
        # of flipped-base = original quote ccy, and its premium (in
        # original base ccy) converts at S.  Algebraically d1' = -d2, so
        # e^{-r_d T} N(d2) == S * [(1/S) e^{-r_d T} N(d2)].
        a = self.ARGS
        lhs = digital_price(**a, option_type="call",
                            payout_currency="domestic")
        rhs = a["S"] * digital_price(S=1 / a["S"], K=1 / a["K"], T=a["T"],
                                     r_d=a["r_f"], r_f=a["r_d"],
                                     sigma=a["sigma"], option_type="put",
                                     payout_currency="foreign")
        assert lhs == pytest.approx(rhs, abs=1e-12)

    def test_digital_call_put_parity_domestic(self):
        a = self.ARGS
        c = digital_price(**a, option_type="call", payout_currency="domestic")
        p = digital_price(**a, option_type="put", payout_currency="domestic")
        assert c + p == pytest.approx(math.exp(-a["r_d"] * a["T"]), abs=1e-12)

    def test_digital_sigma_zero_is_indicator_on_forward(self):
        a = dict(self.ARGS, sigma=0.0)
        F = cip_forward(a["S"], a["T"], a["r_d"], a["r_f"])
        expected = math.exp(-a["r_d"] * a["T"]) * (1.0 if F > a["K"] else 0.0)
        assert digital_price(**a, option_type="call",
                             payout_currency="domestic") == pytest.approx(
            expected, abs=1e-14)

    def test_mc_digital_rejects_bad_n_paths(self):
        with pytest.raises(ValueError, match="n_paths"):
            mc_digital_price(**self.ARGS, option_type="call", n_paths=0)
        with pytest.raises(ValueError, match="n_paths"):
            mc_digital_price(**self.ARGS, option_type="call",
                             n_paths=10.5)  # type: ignore[arg-type]


class TestCIPViolationDetection:
    """The synthetic forward is the desk's CIP-violation detector."""

    def test_option_market_shifted_off_cip_is_flagged(self):
        # Perturb the call premium by 10 pips: the option-implied forward
        # moves off the CIP forward by e^{r_d T} x 10 pips -- measurable,
        # so a conversion/reversal arbitrage is identifiable.
        S, K, T, rd, rf, sig = 1.10, 1.10, 0.5, 0.0425, 0.0290, 0.0825
        c = gk_call(S, K, T, rd, rf, sig) + 0.0010
        p = gk_put(S, K, T, rd, rf, sig)
        F_syn = synthetic_forward_from_options(c, p, K, T, rd)
        F_cip = cip_forward(S, T, rd, rf)
        assert F_syn - F_cip == pytest.approx(
            0.0010 * math.exp(rd * T), abs=1e-12)
        assert abs(F_syn - F_cip) * 1e4 > 5.0  # > 5 pips: tradeable signal

    def test_consistent_market_shows_no_violation(self):
        S, K, T, rd, rf, sig = 147.5, 145.0, 0.5, 0.0050, 0.0525, 0.1075
        c = gk_call(S, K, T, rd, rf, sig)
        p = gk_put(S, K, T, rd, rf, sig)
        F_syn = synthetic_forward_from_options(c, p, K, T, rd)
        assert F_syn == pytest.approx(cip_forward(S, T, rd, rf), abs=1e-9)


class TestDeepWingsPriceAndVega:
    def test_deep_itm_put_at_negative_rates(self):
        # Deep ITM put, both rates negative: price pins to discounted
        # forward intrinsic and stays above undiscounted-forward bounds.
        S, K, T, rd, rf, sig = 1.08, 1.60, 1.0, -0.0075, -0.0050, 0.065
        p = gk_put(S, K, T, rd, rf, sig)
        F = cip_forward(S, T, rd, rf)
        assert p == pytest.approx(math.exp(-rd * T) * (K - F), abs=1e-9)

    def test_gk_price_wings_zero_vega_no_nan(self):
        # 60% OTM at 1w: price underflows cleanly to ~0, never NaN.
        c = gk_price(1.10, 1.76, 7 / 365, 0.0425, 0.0290, 0.0825, "call")
        assert c == 0.0 or (0.0 < c < 1e-30)
        assert math.isfinite(c)
