"""Edge cases from the documentation contract: each is documented in
docs/METHODOLOGY.md / VALIDATION.md and unit-tested here."""

import math

import pytest

from fx_options import (analytic_greeks, binomial_price, delta, gk_call,
                        gk_price, gk_put, implied_vol, mc_price)


class TestNegativeRates:
    """EUR/CHF era: both rates negative must work end to end."""

    ARGS = dict(S=1.08, K=1.08, T=1.0, r_d=-0.0075, r_f=-0.0050,
                sigma=0.065)

    def test_pricing_and_parity(self):
        c = gk_price(**self.ARGS, option_type="call")
        p = gk_price(**self.ARGS, option_type="put")
        assert c > 0 and p > 0
        rhs = (self.ARGS["S"] * math.exp(-self.ARGS["r_f"])
               - self.ARGS["K"] * math.exp(-self.ARGS["r_d"]))
        assert c - p == pytest.approx(rhs, abs=1e-12)

    def test_greeks_and_implied_vol(self):
        g = analytic_greeks(**self.ARGS, option_type="call")
        assert g.vega > 0 and g.gamma > 0
        px = gk_price(**self.ARGS, option_type="call")
        assert implied_vol(px, self.ARGS["S"], self.ARGS["K"],
                           self.ARGS["T"], self.ARGS["r_d"],
                           self.ARGS["r_f"], "call") == pytest.approx(
            self.ARGS["sigma"], abs=1e-8)

    def test_tree_and_delta_conventions(self):
        tree = binomial_price(**self.ARGS, option_type="call", steps=800)
        assert tree == pytest.approx(gk_price(**self.ARGS,
                                              option_type="call"), abs=1e-4)
        for conv in ("spot", "forward", "spot_pa", "forward_pa"):
            d = delta(**self.ARGS, option_type="call", convention=conv)
            assert 0 < d < 1.1  # e^{-r_f T} > 1 with negative r_f


class TestEqualRates:
    def test_rd_equals_rf_forward_equals_spot(self):
        # F = S; GK collapses to zero-drift Black.
        c = gk_call(1.10, 1.10, 0.5, 0.02, 0.02, 0.10)
        p = gk_put(1.10, 1.10, 0.5, 0.02, 0.02, 0.10)
        assert c == pytest.approx(p, abs=1e-14)  # ATM-forward symmetry


class TestExtremeVol:
    def test_very_high_vol_em_pair(self):
        # EM-style 150% vol: price must stay within no-arb bounds.
        S, K, T, rd, rf = 18.5, 20.0, 0.5, 0.1125, 0.045
        c = gk_call(S, K, T, rd, rf, 1.50)
        assert 0 < c < S * math.exp(-rf * T)
        assert implied_vol(c, S, K, T, rd, rf, "call") == pytest.approx(
            1.50, abs=1e-7)

    def test_vol_to_infinity_call_approaches_discounted_spot(self):
        c = gk_call(1.10, 1.10, 1.0, 0.03, 0.01, 8.0)
        assert c == pytest.approx(1.10 * math.exp(-0.01), rel=1e-3)

    def test_vol_monotonicity_preserved_at_extremes(self):
        p1 = gk_call(1.1, 1.1, 0.5, 0.03, 0.01, 1.0)
        p2 = gk_call(1.1, 1.1, 0.5, 0.03, 0.01, 3.0)
        assert p2 > p1


class TestExtremeSpot:
    def test_spot_extremely_small(self):
        c = gk_call(1e-6, 1.0, 0.5, 0.03, 0.01, 0.2)
        p = gk_put(1e-6, 1.0, 0.5, 0.03, 0.01, 0.2)
        assert c == pytest.approx(0.0, abs=1e-12)
        assert p == pytest.approx(math.exp(-0.015) * 1.0, rel=1e-4)

    def test_spot_extremely_large(self):
        c = gk_call(1e6, 1.0, 0.5, 0.03, 0.01, 0.2)
        assert c == pytest.approx(1e6 * math.exp(-0.005), rel=1e-6)
        assert gk_put(1e6, 1.0, 0.5, 0.03, 0.01, 0.2) < 1e-10


class TestTinyTenor:
    def test_t_one_hour(self):
        T = 1.0 / (365 * 24)
        c = gk_call(1.10, 1.10, T, 0.03, 0.01, 0.10)
        assert 0 < c < 0.01
        assert implied_vol(c, 1.10, 1.10, T, 0.03, 0.01,
                           "call") == pytest.approx(0.10, abs=1e-8)

    def test_t_zero_greeks_raise_not_nan(self):
        with pytest.raises(ValueError):
            analytic_greeks(1.1, 1.1, 0.0, 0.03, 0.01, 0.1, "call")


class TestCrossFunctionValidation:
    @pytest.mark.parametrize("func,kwargs", [
        (gk_price, dict(S=1.1, K=1.1, T=0.5, r_d=0.03, r_f=0.01,
                        sigma=0.1, option_type="swaption")),
        (mc_price, dict(S=1.1, K=1.1, T=0.5, r_d=0.03, r_f=0.01,
                        sigma=-0.1, option_type="call")),
        (binomial_price, dict(S=1.1, K=0.0, T=0.5, r_d=0.03, r_f=0.01,
                              sigma=0.1, option_type="call")),
        (delta, dict(S=1.1, K=1.1, T=0.5, r_d=0.03, r_f=0.01, sigma=0.1,
                     option_type="call", convention="premium")),
    ])
    def test_informative_value_errors(self, func, kwargs):
        with pytest.raises(ValueError):
            func(**kwargs)

    def test_nan_and_inf_rejected_everywhere(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(ValueError):
                gk_call(bad, 1.1, 0.5, 0.03, 0.01, 0.1)
            with pytest.raises(ValueError):
                gk_call(1.1, 1.1, 0.5, bad, 0.01, 0.1)

    def test_bool_rejected_as_input(self):
        with pytest.raises(ValueError):
            gk_call(True, 1.1, 0.5, 0.03, 0.01, 0.1)
