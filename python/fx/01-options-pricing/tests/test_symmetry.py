"""Foreign-domestic symmetry (notional duality).

A EURUSD call (right to buy EUR paying USD) is, viewed from the EUR side,
a USDEUR put (right to sell USD receiving EUR).  Formally:

    C_d(S, K, T, r_d, r_f, sigma) = S * K * P_f(1/S, 1/K, T, r_f, r_d, sigma)

where the flipped option is priced with the rate roles swapped, its
premium expressed in *foreign* currency and rescaled by S*K.
"""

import itertools

import pytest

from fx_options import delta, gk_price

SPOTS = [0.65, 1.10, 147.5]
STRIKE_MULTS = [0.85, 1.0, 1.20]
TENORS = [0.1, 0.5, 2.0]
RATES = [(0.0425, 0.0290), (0.0050, 0.0525), (-0.0075, -0.0050)]


class TestForeignDomesticSymmetry:
    @pytest.mark.parametrize("S,k_mult,T,rd_rf", [
        (s, m, t, r) for s, m, t, r in itertools.product(
            SPOTS, STRIKE_MULTS, TENORS, RATES)
    ])
    def test_call_equals_flipped_put_across_grid(self, S, k_mult, T, rd_rf):
        rd, rf = rd_rf
        K = S * k_mult
        sigma = 0.11
        lhs = gk_price(S, K, T, rd, rf, sigma, "call")
        rhs = S * K * gk_price(1 / S, 1 / K, T, rf, rd, sigma, "put")
        assert lhs == pytest.approx(rhs, abs=1e-10 * max(1.0, S * K))

    def test_put_equals_flipped_call(self):
        S, K, T, rd, rf, sigma = 1.10, 1.05, 0.75, 0.03, 0.01, 0.09
        lhs = gk_price(S, K, T, rd, rf, sigma, "put")
        rhs = S * K * gk_price(1 / S, 1 / K, T, rf, rd, sigma, "call")
        assert lhs == pytest.approx(rhs, abs=1e-12)

    def test_premium_adjusted_delta_is_flipped_forward_delta(self):
        # The PA forward delta of a call is minus the (unadjusted) forward
        # delta of the flipped put, rescaled by the K/F notional
        # conversion: PA call delta = (K/F) N(d2), flipped put forward
        # delta = -N(d2).  This is *why* PA deltas exist: they are the
        # hedge seen from the other currency's viewpoint.
        import math
        S, K, T, rd, rf, sigma = 1.10, 1.15, 0.5, 0.0425, 0.0290, 0.0825
        F = S * math.exp((rd - rf) * T)
        pa = delta(S, K, T, rd, rf, sigma, "call", "forward_pa")
        flipped = delta(1 / S, 1 / K, T, rf, rd, sigma, "put", "forward")
        assert pa == pytest.approx(-(K / F) * flipped, abs=1e-12)
