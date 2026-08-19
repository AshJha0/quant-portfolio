"""NaN/Inf hardening on the secondary API surface, plus arbitrage properties.

The primary pricers (:mod:`fx_options.garman_kohlhagen`, :mod:`black76`,
:mod:`binomial`) already funnel every market input through
:func:`fx_options._common.validate_inputs`, which rejects non-finite values.
The helpers exercised here take *extra* scalar arguments that bypass that
funnel — ``pip_factor``, ``pip_size``, ``transaction_cost_pips``, ``mu``,
``sigma_hedge``, an already-computed ``delta``/``price`` — and a guard
written as ``if x <= 0: raise`` silently accepts NaN because every
comparison with NaN is False.  Each test below pins the fixed behaviour:
non-finite in, ``ValueError`` out, never a NaN "price".

The second half adds arbitrage-boundary properties (calendar monotonicity,
delta bounds, digital bounds) that the analytic-identity suites do not
cover.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fx_options import (
    binomial_price,
    delta,
    digital_price,
    forward_points,
    forward_to_spot_delta,
    gk_price,
    premium_adjust_spot_delta,
    simulate_delta_hedge,
    spot_to_forward_delta,
)
from fx_options.data.synthetic import synthetic_vol_quotes

NON_FINITE = [float("nan"), float("inf"), float("-inf")]

# EURUSD-style base case: spot 1.10, 1y, USD 4%, EUR 2.5%, 9 vol.
MKT = dict(S=1.10, K=1.10, T=1.0, r_d=0.04, r_f=0.025, sigma=0.09)


class TestForwardPointsPipFactor:
    """``pip_factor`` is a bare scalar, not routed through validate_inputs."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_pip_factor_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="pip_factor"):
            forward_points(1.10, 1.0, 0.04, 0.025, pip_factor=bad)

    def test_zero_and_negative_pip_factor_still_raise(self) -> None:
        for bad in (0.0, -1e4):
            with pytest.raises(ValueError, match="pip_factor"):
                forward_points(1.10, 1.0, 0.04, 0.025, pip_factor=bad)

    def test_jpy_pip_factor_is_100x_smaller_than_default(self) -> None:
        # Same market, different quoting convention: USDJPY pips are 1e-2.
        pts_4dp = forward_points(150.0, 1.0, 0.001, 0.04, pip_factor=1e4)
        pts_2dp = forward_points(150.0, 1.0, 0.001, 0.04, pip_factor=1e2)
        assert pts_2dp == pytest.approx(pts_4dp / 100.0, rel=1e-12)
        # USD (foreign here) yields more than JPY (domestic) -> forward discount.
        assert pts_2dp < 0.0


class TestDeltaHelperHardening:
    """Delta conversions accept a pre-computed delta; NaN must not survive."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_spot_to_forward_rejects_non_finite_delta(self, bad: float) -> None:
        with pytest.raises(ValueError, match="delta_spot"):
            spot_to_forward_delta(bad, 1.0, 0.025)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_forward_to_spot_rejects_non_finite_delta(self, bad: float) -> None:
        with pytest.raises(ValueError, match="delta_forward"):
            forward_to_spot_delta(bad, 1.0, 0.025)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_premium_adjust_rejects_non_finite_premium(self, bad: float) -> None:
        with pytest.raises(ValueError, match="price"):
            premium_adjust_spot_delta(0.55, bad, 1.10)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_premium_adjust_rejects_non_finite_delta(self, bad: float) -> None:
        with pytest.raises(ValueError, match="delta_spot"):
            premium_adjust_spot_delta(bad, 0.04, 1.10)

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -1.10])
    def test_premium_adjust_rejects_bad_spot(self, bad: float) -> None:
        with pytest.raises(ValueError, match="Spot S"):
            premium_adjust_spot_delta(0.55, 0.04, bad)

    def test_premium_adjust_matches_the_spot_pa_convention(self) -> None:
        # delta_pa = delta_spot - V/S is the identity the desk uses to
        # re-express a hedge when the premium settles in base currency.
        d_spot = delta(**MKT, option_type="call", convention="spot")
        prem = gk_price(**MKT, option_type="call")
        manual = premium_adjust_spot_delta(d_spot, prem, MKT["S"])
        closed = delta(**MKT, option_type="call", convention="spot_pa")
        assert manual == pytest.approx(closed, abs=1e-12)


class TestHedgeSimulatorHardening:
    """The hedge simulator takes five scalars outside validate_inputs."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_sigma_hedge_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="sigma_hedge"):
            simulate_delta_hedge(1.10, 1.10, 0.25, 0.04, 0.025, 0.09, "call",
                                 sigma_hedge=bad, n_rebalances=4, n_paths=8)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_drift_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="mu"):
            simulate_delta_hedge(1.10, 1.10, 0.25, 0.04, 0.025, 0.09, "call",
                                 mu=bad, n_rebalances=4, n_paths=8)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_transaction_cost_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="transaction_cost_pips"):
            simulate_delta_hedge(1.10, 1.10, 0.25, 0.04, 0.025, 0.09, "call",
                                 transaction_cost_pips=bad,
                                 n_rebalances=4, n_paths=8)

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -1e-4])
    def test_bad_pip_size_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="pip_size"):
            simulate_delta_hedge(1.10, 1.10, 0.25, 0.04, 0.025, 0.09, "call",
                                 pip_size=bad, n_rebalances=4, n_paths=8)

    def test_finite_run_produces_no_nan_pnl(self) -> None:
        res = simulate_delta_hedge(1.10, 1.10, 0.25, 0.04, 0.025, 0.09,
                                   "call", n_rebalances=10, n_paths=200,
                                   rng=7)
        assert np.all(np.isfinite(res.pnl))
        assert math.isfinite(res.mean_pnl) and math.isfinite(res.std_pnl)


class TestSyntheticQuoteHardening:
    @pytest.mark.parametrize(
        "kwargs, field",
        [
            ({"base_atm": float("nan")}, "base_atm"),
            ({"skew": float("inf")}, "skew"),
            ({"smile": float("nan")}, "smile"),
            ({"noise": float("nan")}, "noise"),
        ],
    )
    def test_non_finite_quote_parameters_raise(self, kwargs, field) -> None:
        with pytest.raises(ValueError, match=field):
            synthetic_vol_quotes(**kwargs)

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -0.25])
    def test_bad_tenor_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="tenors"):
            synthetic_vol_quotes(tenors=(0.25, bad))


class TestArbitrageBoundaries:
    """Static no-arbitrage properties not pinned by the identity suites."""

    def test_call_value_is_non_decreasing_in_expiry_when_rates_equal(
            self) -> None:
        # With r_d == r_f the forward equals spot, so the discounted payoff
        # process is a submartingale and calendar spreads cannot be
        # negative: C(T2) >= C(T1) for T2 > T1.
        prices = [gk_price(1.10, 1.15, t, 0.02, 0.02, 0.10, "call")
                  for t in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)]
        assert all(b >= a - 1e-14 for a, b in zip(prices, prices[1:]))

    def test_spot_delta_bounded_by_foreign_discount_factor(self) -> None:
        # |delta_spot| <= e^{-r_f T}: the hedge can never be larger than the
        # PV of one unit of base currency.
        for K in (0.5, 0.9, 1.10, 1.4, 3.0):
            for opt in ("call", "put"):
                d = delta(1.10, K, 1.0, 0.04, 0.025, 0.09, opt, "spot")
                assert abs(d) <= math.exp(-0.025 * 1.0) + 1e-12

    def test_domestic_digital_bounded_by_domestic_discount_factor(self) -> None:
        df_d = math.exp(-0.04 * 1.0)
        for K in (0.6, 1.10, 2.5):
            for opt in ("call", "put"):
                v = digital_price(1.10, K, 1.0, 0.04, 0.025, 0.09, opt,
                                  payout_currency="domestic")
                assert -1e-15 <= v <= df_d + 1e-12

    def test_foreign_digital_bounded_by_pv_of_one_base_unit(self) -> None:
        cap = 1.10 * math.exp(-0.025 * 1.0)
        for K in (0.6, 1.10, 2.5):
            for opt in ("call", "put"):
                v = digital_price(1.10, K, 1.0, 0.04, 0.025, 0.09, opt,
                                  payout_currency="foreign")
                assert -1e-15 <= v <= cap + 1e-12

    def test_binomial_rejects_step_size_that_breaks_the_probability(
            self) -> None:
        # A managed/pegged pair: near-zero vol with a wide rate differential
        # pushes p = (e^{(r_d-r_f)dt} - d)/(u - d) outside [0, 1] unless dt
        # is small.  The tree must refuse rather than return a bogus price.
        with pytest.raises(ValueError, match="outside"):
            binomial_price(7.80, 7.80, 1.0, 0.05, 0.001, 0.002, "call",
                           steps=2)
        # With enough steps the same trade prices cleanly and finitely.
        ok = binomial_price(7.80, 7.80, 1.0, 0.05, 0.001, 0.002, "call",
                            steps=4000)
        assert math.isfinite(ok) and ok > 0.0
