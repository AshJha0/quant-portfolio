"""Garman-Kohlhagen: textbook values, parity, BS reduction, implied vol."""

import math

import pytest
from scipy.stats import norm

from fx_options import d1, d2, gk_call, gk_price, gk_put, implied_vol

# A representative grid: (S, K, T, r_d, r_f, sigma)
GRID = [
    (1.10, 1.10, 0.5, 0.0425, 0.0290, 0.0825),
    (1.10, 1.00, 0.25, 0.0425, 0.0290, 0.0825),
    (1.10, 1.25, 1.0, 0.0425, 0.0290, 0.0825),
    (147.5, 147.5, 0.5, 0.0050, 0.0525, 0.1075),
    (147.5, 130.0, 2.0, 0.0050, 0.0525, 0.1075),
    (1.08, 1.08, 1.0, -0.0075, -0.0050, 0.065),
    (18.5, 20.0, 0.25, 0.1125, 0.045, 0.35),
]


class TestTextbookValues:
    def test_haug_currency_call(self):
        # Haug, "Complete Guide to Option Pricing Formulas": USD call on
        # foreign ccy, S=1.56, K=1.60, T=0.5, r_d=6%, r_f=8%, sigma=12%.
        assert gk_call(1.56, 1.60, 0.5, 0.06, 0.08, 0.12) == pytest.approx(
            0.0291, abs=1e-4)

    def test_hull_gbp_call(self):
        # Hull, "Options, Futures and Other Derivatives": 4m call on GBP,
        # S=1.6000, K=1.6000, r_d=8%, r_f=11%, sigma=14.1% -> 0.0430.
        assert gk_call(1.60, 1.60, 4 / 12, 0.08, 0.11, 0.141) == pytest.approx(
            0.0430, abs=1e-4)

    def test_d1_d2_relation(self):
        S, K, T, rd, rf, sig = GRID[0]
        assert d2(S, K, T, rd, rf, sig) == pytest.approx(
            d1(S, K, T, rd, rf, sig) - sig * math.sqrt(T), abs=1e-14)


class TestParityAndReduction:
    @pytest.mark.parametrize("S,K,T,rd,rf,sig", GRID)
    def test_put_call_parity_two_rates(self, S, K, T, rd, rf, sig):
        # C - P = S e^{-r_f T} - K e^{-r_d T}, exact to 1e-10 (scaled for
        # JPY-level spots by using relative tolerance on large notionals).
        c = gk_call(S, K, T, rd, rf, sig)
        p = gk_put(S, K, T, rd, rf, sig)
        lhs = c - p
        rhs = S * math.exp(-rf * T) - K * math.exp(-rd * T)
        assert lhs == pytest.approx(rhs, abs=1e-10 * max(1.0, S))

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_gk_equals_bs_with_dividend_yield(self, option_type):
        # Independent Black-Scholes-with-yield implementation, q = r_f.
        S, K, T, r, q, sig = 1.10, 1.15, 0.75, 0.03, 0.015, 0.10
        phi = 1.0 if option_type == "call" else -1.0
        _d1 = (math.log(S / K) + (r - q + sig**2 / 2) * T) / (sig * math.sqrt(T))
        _d2 = _d1 - sig * math.sqrt(T)
        bs = phi * (S * math.exp(-q * T) * norm.cdf(phi * _d1)
                    - K * math.exp(-r * T) * norm.cdf(phi * _d2))
        assert gk_price(S, K, T, r, q, sig, option_type) == pytest.approx(
            bs, abs=1e-14)

    def test_price_vs_lognormal_quadrature(self):
        # Cross-check against direct numerical integration of the payoff
        # over the risk-neutral lognormal density.
        from scipy.integrate import quad
        S, K, T, rd, rf, sig = 1.10, 1.14, 0.5, 0.04, 0.01, 0.12
        mu = math.log(S) + (rd - rf - sig**2 / 2) * T
        sd = sig * math.sqrt(T)

        def integrand(x):
            return max(math.exp(x) - K, 0.0) * norm.pdf(x, mu, sd)

        val, _ = quad(integrand, math.log(K), mu + 12 * sd, limit=200)
        assert gk_call(S, K, T, rd, rf, sig) == pytest.approx(
            math.exp(-rd * T) * val, abs=1e-10)


class TestProperties:
    def test_monotone_increasing_in_vol(self):
        prices = [gk_call(1.1, 1.1, 0.5, 0.02, 0.01, s)
                  for s in (0.05, 0.10, 0.20, 0.40, 0.80)]
        assert all(b > a for a, b in zip(prices, prices[1:]))

    def test_convex_in_strike(self):
        ks = [1.00, 1.05, 1.10, 1.15, 1.20]
        prices = [gk_call(1.1, k, 0.5, 0.02, 0.01, 0.10) for k in ks]
        for i in range(1, len(ks) - 1):
            assert prices[i] <= 0.5 * (prices[i - 1] + prices[i + 1]) + 1e-14

    def test_call_within_no_arbitrage_bounds(self):
        S, K, T, rd, rf, sig = 1.1, 1.1, 0.5, 0.03, 0.01, 0.10
        c = gk_call(S, K, T, rd, rf, sig)
        F = S * math.exp((rd - rf) * T)
        assert math.exp(-rd * T) * max(F - K, 0.0) <= c <= S * math.exp(-rf * T)


class TestLimits:
    def test_t_zero_intrinsic(self):
        assert gk_call(1.2, 1.1, 0.0, 0.03, 0.01, 0.10) == pytest.approx(0.1)
        assert gk_put(1.2, 1.1, 0.0, 0.03, 0.01, 0.10) == 0.0

    def test_sigma_zero_discounted_forward_intrinsic(self):
        S, K, T, rd, rf = 1.10, 1.10, 1.0, 0.05, 0.01
        F = S * math.exp((rd - rf) * T)
        expected = math.exp(-rd * T) * (F - K)
        assert gk_call(S, K, T, rd, rf, 0.0) == pytest.approx(expected, abs=1e-14)
        assert gk_put(S, K, T, rd, rf, 0.0) == 0.0

    def test_deep_itm_approaches_forward_intrinsic(self):
        S, K, T, rd, rf, sig = 1.1, 0.2, 0.5, 0.03, 0.01, 0.10
        F = S * math.exp((rd - rf) * T)
        assert gk_call(S, K, T, rd, rf, sig) == pytest.approx(
            math.exp(-rd * T) * (F - K), abs=1e-12)

    def test_deep_otm_call_near_zero(self):
        assert gk_call(1.1, 5.0, 0.25, 0.03, 0.01, 0.10) < 1e-12


class TestImpliedVol:
    # Note: at T = 0.05 with strikes 15-20% away the time value falls
    # below double-precision resolution of the intrinsic (~1e-17), so the
    # vol is fundamentally unrecoverable — documented as a failure mode in
    # docs/VALIDATION.md; the grid below stays inside the recoverable zone.
    @pytest.mark.parametrize("k_mult", [0.85, 0.95, 1.0, 1.05, 1.20])
    @pytest.mark.parametrize("T", [0.25, 0.5, 2.0])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_round_trip_grid(self, k_mult, T, option_type):
        S, rd, rf, sig = 1.10, 0.0425, 0.0290, 0.0825
        K = S * k_mult
        px = gk_price(S, K, T, rd, rf, sig, option_type)
        iv = implied_vol(px, S, K, T, rd, rf, option_type)
        assert iv == pytest.approx(sig, abs=1e-8)

    def test_high_vol_round_trip(self):
        px = gk_call(18.5, 22.0, 0.25, 0.1125, 0.045, 0.60)
        assert implied_vol(px, 18.5, 22.0, 0.25, 0.1125, 0.045,
                           "call") == pytest.approx(0.60, abs=1e-8)

    def test_price_below_intrinsic_raises(self):
        with pytest.raises(ValueError, match="no-arbitrage"):
            implied_vol(0.001, 1.3, 1.1, 0.5, 0.03, 0.01, "call")

    def test_price_above_upper_bound_raises(self):
        with pytest.raises(ValueError, match="no-arbitrage"):
            implied_vol(2.0, 1.1, 1.1, 0.5, 0.03, 0.01, "call")

    def test_t_zero_raises(self):
        with pytest.raises(ValueError, match="T > 0"):
            implied_vol(0.05, 1.1, 1.1, 0.0, 0.03, 0.01, "call")

    def test_long_dated_deep_itm_high_vol_flat_plateau_raises(self):
        """Regression: deep ITM + long-dated + high vol drives |d1|, |d2|
        large enough that N(d1)/N(d2) saturate to 0/1 in double precision,
        so gk_price(sigma) is bit-identical to the sigma->inf bound for
        every sigma from the true root up to whatever `hi` the Brent
        bracket-expansion loop happens to stop at. Before the fix this
        silently returned that arbitrary `hi` (e.g. 4.0 when the true vol
        was 3.0 -- a 33% relative error with no signal to the caller);
        the solver must now recognise the flat plateau and raise instead
        of guessing.
        """
        S, K, T, r_d, r_f, sigma = 1.0, 0.5, 30.0, 0.03, 0.01, 3.0
        price = gk_call(S, K, T, r_d, r_f, sigma)
        with pytest.raises(ValueError, match="unrecoverably large"):
            implied_vol(price, S, K, T, r_d, r_f, "call")


class TestValidation:
    @pytest.mark.parametrize("bad", [
        dict(S=-1.0), dict(S=0.0), dict(K=-2.0), dict(T=-0.1),
        dict(sigma=-0.2), dict(S=float("nan")), dict(r_d=float("inf")),
    ])
    def test_invalid_inputs_raise(self, bad):
        kwargs = dict(S=1.1, K=1.1, T=0.5, r_d=0.03, r_f=0.01, sigma=0.1)
        kwargs.update(bad)
        with pytest.raises(ValueError):
            gk_price(**kwargs, option_type="call")

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            gk_price(1.1, 1.1, 0.5, 0.03, 0.01, 0.1, "straddle")

    def test_d1_undefined_at_zero_vol(self):
        with pytest.raises(ValueError, match="d1 undefined"):
            d1(1.1, 1.1, 0.5, 0.03, 0.01, 0.0)
