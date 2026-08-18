"""Forwards (CIP) and Black-76 on the FX forward."""

import math

import pytest

from fx_options import (black76_from_spot, black76_price, cip_forward,
                        forward_points, gk_call, gk_price, gk_put,
                        synthetic_forward_from_options)

GRID = [
    (1.10, 1.10, 0.5, 0.0425, 0.0290, 0.0825),
    (1.10, 1.02, 0.25, 0.0425, 0.0290, 0.0825),
    (147.5, 150.0, 0.5, 0.0050, 0.0525, 0.1075),
    (1.08, 1.10, 1.0, -0.0075, -0.0050, 0.065),
    (18.5, 20.0, 0.25, 0.1125, 0.045, 0.35),
]


class TestCIPForward:
    def test_value(self):
        F = cip_forward(1.10, 0.5, 0.0425, 0.0290)
        assert F == pytest.approx(1.10 * math.exp(0.0135 * 0.5), abs=1e-14)

    def test_forward_premium_when_rd_above_rf(self):
        assert cip_forward(1.10, 1.0, 0.05, 0.02) > 1.10

    def test_forward_discount_when_rf_above_rd(self):
        # USDJPY-style carry: base ccy (USD) yields more -> forward < spot.
        assert cip_forward(147.5, 1.0, 0.005, 0.0525) < 147.5

    def test_zero_tenor(self):
        assert cip_forward(1.10, 0.0, 0.05, 0.02) == 1.10

    def test_forward_points_pip_factors(self):
        pts_eur = forward_points(1.10, 0.5, 0.0425, 0.0290, pip_factor=1e4)
        pts_jpy = forward_points(147.5, 0.5, 0.0050, 0.0525, pip_factor=1e2)
        assert pts_eur == pytest.approx(
            (cip_forward(1.10, 0.5, 0.0425, 0.0290) - 1.10) * 1e4)
        assert pts_jpy < 0  # discount on USD vs JPY

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            cip_forward(-1.0, 0.5, 0.03, 0.01)
        with pytest.raises(ValueError):
            cip_forward(1.1, -0.5, 0.03, 0.01)
        with pytest.raises(ValueError, match="pip_factor"):
            forward_points(1.1, 0.5, 0.03, 0.01, pip_factor=0.0)


class TestSyntheticForward:
    @pytest.mark.parametrize("S,K,T,rd,rf,sig", GRID)
    def test_option_implied_forward_matches_cip(self, S, K, T, rd, rf, sig):
        c = gk_call(S, K, T, rd, rf, sig)
        p = gk_put(S, K, T, rd, rf, sig)
        F_syn = synthetic_forward_from_options(c, p, K, T, rd)
        assert F_syn == pytest.approx(cip_forward(S, T, rd, rf),
                                      abs=1e-10 * max(1.0, S))

    def test_invalid_premium_raises(self):
        with pytest.raises(ValueError, match="call_price"):
            synthetic_forward_from_options(float("nan"), 0.01, 1.1, 0.5, 0.03)


class TestBlack76:
    @pytest.mark.parametrize("S,K,T,rd,rf,sig", GRID)
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_equals_gk_to_1e10(self, S, K, T, rd, rf, sig, option_type):
        b76 = black76_from_spot(S, K, T, rd, rf, sig, option_type)
        gk = gk_price(S, K, T, rd, rf, sig, option_type)
        assert b76 == pytest.approx(gk, abs=1e-10)

    def test_direct_forward_input(self):
        F, K, T, rd, sig = 1.1075, 1.10, 0.5, 0.0425, 0.0825
        c = black76_price(F, K, T, rd, sig, "call")
        p = black76_price(F, K, T, rd, sig, "put")
        # Black-76 parity: C - P = e^{-r_d T}(F - K).
        assert c - p == pytest.approx(math.exp(-rd * T) * (F - K), abs=1e-14)

    def test_t_zero_and_sigma_zero(self):
        assert black76_price(1.2, 1.1, 0.0, 0.03, 0.1, "call") == pytest.approx(0.1)
        assert black76_price(1.2, 1.1, 0.5, 0.03, 0.0, "call") == pytest.approx(
            math.exp(-0.015) * 0.1, abs=1e-14)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            black76_price(-1.0, 1.1, 0.5, 0.03, 0.1, "call")
        with pytest.raises(ValueError):
            black76_price(1.1, 1.1, 0.5, 0.03, -0.1, "call")
        with pytest.raises(ValueError):
            black76_price(1.1, 1.1, 0.5, 0.03, 0.1, "digital")
