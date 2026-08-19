"""Non-finite inputs must raise, not propagate into curves, forwards and risk.

Almost every guard in `fx_rates` was written as an inequality —
``if spot <= 0.0: raise``, ``if tau <= 0.0: raise``,
``if bid > ask: raise``, ``if strike <= 0.0: raise``. Every comparison
against NaN is False, so each of those *accepted* NaN and the NaN then flowed
silently through the discount curve, the CIP forward, the mark-to-market and
the DV01 ladder.

The sharpest case was :func:`fx_rates.detect_cip_arbitrage`: with a NaN in
the two-sided quotes, the crossed-market check (``bid > ask``), the
positivity check and the final ``best_pnl > min_pnl`` comparison were all
False, so the detector returned a confident **"no arbitrage"** on data it
could not price at all. A missed arbitrage signal that looks like a clean
result is a worse outcome than an exception.

These tests pin the corrected behaviour and, in each class, keep at least one
positive test proving the finite path is untouched.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from fx_rates import (
    CIPQuotes,
    DiscountCurve,
    FXForward,
    FXSwap,
    MarketState,
    Scenario,
    basis_adjusted_curve,
    cip_forward,
    deposit_rate_from_df,
    detect_cip_arbitrage,
    df_from_deposit,
    dv01,
    forward_carry,
    forward_points,
    fx_delta,
    implied_basis_from_forwards,
    market_forward,
)
from fx_rates.xccy import CrossCurrencySwap

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.fixture(scope="module")
def usd_curve() -> DiscountCurve:
    t = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
    return DiscountCurve.from_zero_rates(t, np.array([0.052, 0.051, 0.048,
                                                      0.044, 0.041]), "USD")


@pytest.fixture(scope="module")
def eur_curve() -> DiscountCurve:
    t = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
    return DiscountCurve.from_zero_rates(t, np.array([0.038, 0.037, 0.034,
                                                      0.030, 0.028]), "EUR")


@pytest.fixture(scope="module")
def market(usd_curve, eur_curve) -> MarketState:
    return MarketState(1.0850, usd_curve, eur_curve,
                       ((1.0, -0.0012), (5.0, -0.0025)), ("EUR", "USD"))


def good_quotes(**overrides) -> dict:
    """A clean, non-arbitrageable two-sided EURUSD 3m market."""
    q = dict(spot_bid=1.0849, spot_ask=1.0851,
             fwd_bid=1.0885, fwd_ask=1.0887,
             dom_rate_bid=0.0520, dom_rate_ask=0.0530,
             for_rate_bid=0.0370, for_rate_ask=0.0380,
             tau=0.25)
    q.update(overrides)
    return q


class TestCipArbitrageDetectorGuards:
    """The headline defect: NaN quotes silently reported 'no arbitrage'."""

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize(
        "field",
        ["spot_bid", "spot_ask", "fwd_bid", "fwd_ask", "dom_rate_bid",
         "dom_rate_ask", "for_rate_bid", "for_rate_ask", "tau"],
    )
    def test_non_finite_quote_rejected(self, field, bad) -> None:
        with pytest.raises(ValueError, match=field):
            CIPQuotes(**good_quotes(**{field: bad}))

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_min_pnl_rejected(self, bad) -> None:
        q = CIPQuotes(**good_quotes())
        with pytest.raises(ValueError, match="min_pnl"):
            detect_cip_arbitrage(q, min_pnl=bad)

    def test_clean_market_still_reports_no_arbitrage(self) -> None:
        res = detect_cip_arbitrage(CIPQuotes(**good_quotes()))
        assert res.is_arbitrage is False
        assert res.direction == "none"
        assert np.isfinite(res.f_lower) and np.isfinite(res.f_upper)
        assert res.f_lower < res.f_upper

    def test_genuine_violation_is_still_detected(self) -> None:
        # Push the forward far above the upper bound: sell-forward arbitrage.
        q = CIPQuotes(**good_quotes(fwd_bid=1.20, fwd_ask=1.2002))
        res = detect_cip_arbitrage(q)
        assert res.is_arbitrage is True
        assert res.direction == "sell_forward"
        assert res.pnl > 0.0 and np.isfinite(res.pnl)


class TestCurveQueryGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_df_at_non_finite_time_rejected(self, usd_curve, bad) -> None:
        with pytest.raises(ValueError, match="non-finite time"):
            usd_curve.df(bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_zero_rate_at_non_finite_time_rejected(self, usd_curve,
                                                   bad) -> None:
        with pytest.raises(ValueError, match="non-finite time"):
            usd_curve.zero_rate(bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_time_inside_an_array_rejected(self, usd_curve,
                                                      bad) -> None:
        with pytest.raises(ValueError, match="non-finite time"):
            usd_curve.df(np.array([0.5, bad, 2.0]))

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_parallel_shift_rejected(self, usd_curve, bad) -> None:
        with pytest.raises(ValueError, match="finite"):
            usd_curve.parallel_shift(bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_pillar_shift_rejected(self, usd_curve, bad) -> None:
        with pytest.raises(ValueError, match="finite"):
            usd_curve.pillar_shift(1, bad)

    def test_finite_queries_are_unchanged(self, usd_curve) -> None:
        assert usd_curve.df(0.0) == pytest.approx(1.0, abs=1e-15)
        assert usd_curve.zero_rate(1.0) == pytest.approx(0.048, rel=1e-12)
        # A +1bp parallel shift lowers every discount factor.
        up = usd_curve.parallel_shift(1.0)
        assert np.all(up.dfs < usd_curve.dfs)


class TestMarketAndForwardGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_spot_rejected(self, usd_curve, eur_curve, bad) -> None:
        with pytest.raises(ValueError, match="spot"):
            MarketState(bad, usd_curve, eur_curve, (), ("EUR", "USD"))

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_basis_spread_rejected(self, usd_curve, eur_curve,
                                              bad) -> None:
        with pytest.raises(ValueError, match="basis"):
            MarketState(1.085, usd_curve, eur_curve, ((1.0, bad),),
                        ("EUR", "USD"))

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_cip_forward_rejects_non_finite_spot(self, usd_curve, eur_curve,
                                                 bad) -> None:
        with pytest.raises(ValueError, match="spot"):
            cip_forward(bad, usd_curve, eur_curve, 1.0)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_cip_forward_rejects_non_finite_expiry(self, usd_curve, eur_curve,
                                                   bad) -> None:
        with pytest.raises(ValueError, match="expiry|non-finite time"):
            cip_forward(1.085, usd_curve, eur_curve, bad)

    @pytest.mark.parametrize("bad", NON_FINITE + [0.0, -1e4])
    def test_forward_points_rejects_bad_point_factor(self, market,
                                                     bad) -> None:
        with pytest.raises(ValueError, match="point_factor"):
            forward_points(market, 1.0, point_factor=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_non_finite_basis_curve_input_rejected(self, eur_curve,
                                                   bad) -> None:
        with pytest.raises(ValueError, match="basis"):
            basis_adjusted_curve(eur_curve, ((1.0, -0.001), (5.0, bad)))

    def test_cip_and_market_forwards_still_agree_up_to_the_basis(
            self, market) -> None:
        f_cip = cip_forward(market.spot, market.domestic_curve,
                            market.foreign_curve, 5.0)
        f_mkt = market_forward(market, 5.0)
        # EURUSD basis is negative -> the basis-adjusted foreign curve
        # discounts less -> the market forward sits *above* pure CIP.
        assert f_mkt > f_cip
        implied = -np.log(f_mkt / f_cip) / 5.0
        assert implied == pytest.approx(-0.0025, rel=1e-9)


class TestInstrumentConstructionGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize("field", ["notional_base", "strike", "expiry"])
    def test_fx_forward_fields(self, field, bad) -> None:
        kwargs = {"notional_base": 10e6, "strike": 1.09, "expiry": 1.0}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            FXForward(**kwargs)

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize(
        "field",
        ["notional_base", "near_strike", "near_expiry", "far_strike",
         "far_expiry"],
    )
    def test_fx_swap_fields(self, field, bad) -> None:
        kwargs = {"notional_base": 10e6, "near_strike": 1.085,
                  "near_expiry": 0.25, "far_strike": 1.09, "far_expiry": 1.0}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            FXSwap(**kwargs)

    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize(
        "field",
        ["notional_base", "notional_quote", "rate_base", "rate_quote",
         "maturity"],
    )
    def test_xccy_swap_fields(self, field, bad) -> None:
        kwargs = {"notional_base": 10e6, "notional_quote": 10.85e6,
                  "rate_base": 0.031, "rate_quote": 0.047, "maturity": 5.0}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            CrossCurrencySwap(**kwargs)

    def test_valid_instruments_price_finitely(self, market) -> None:
        fwd = FXForward(10e6, 1.09, 1.0)
        swap = FXSwap(10e6, 1.085, 0.25, 1.09, 1.0)
        for v in (fwd.value(market), swap.value(market)):
            assert np.isfinite(v)
        # The two valuation routes for the outright agree analytically.
        assert fwd.value(market, "cashflows") == pytest.approx(
            fwd.value(market, "forward"), rel=1e-10)


class TestDepositAndBasisHelperGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_df_from_deposit_rejects_non_finite_rate(self, bad) -> None:
        with pytest.raises(ValueError, match="rate"):
            df_from_deposit(bad, 0.25)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_df_from_deposit_rejects_non_finite_tau(self, bad) -> None:
        with pytest.raises(ValueError, match="tau"):
            df_from_deposit(0.05, bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_deposit_rate_from_df_rejects_non_finite(self, bad) -> None:
        with pytest.raises(ValueError, match="df|tau"):
            deposit_rate_from_df(bad, 0.25)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_implied_basis_rejects_non_finite_spot(self, usd_curve,
                                                   eur_curve, bad) -> None:
        with pytest.raises(ValueError, match="spot"):
            implied_basis_from_forwards(bad, usd_curve, eur_curve,
                                        [(1.0, 1.09)])

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_implied_basis_rejects_non_finite_quote(self, usd_curve,
                                                    eur_curve, bad) -> None:
        with pytest.raises(ValueError, match="forward_quote|forward_tenor"):
            implied_basis_from_forwards(1.085, usd_curve, eur_curve,
                                        [(1.0, bad)])

    def test_deposit_identity_round_trips_including_negative_rates(
            self) -> None:
        for rate in (-0.0075, -0.001, 0.0, 0.052):
            df = df_from_deposit(rate, 0.25)
            assert deposit_rate_from_df(df, 0.25) == pytest.approx(rate,
                                                                   abs=1e-14)


class TestScenarioAndRiskGuards:
    @pytest.mark.parametrize("bad", NON_FINITE)
    @pytest.mark.parametrize(
        "field", ["spot_pct", "domestic_bp", "foreign_bp", "basis_bp"])
    def test_non_finite_scenario_shock_rejected(self, field, bad) -> None:
        kwargs = {"name": "bad"}
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            Scenario(**kwargs)

    def test_scenario_below_minus_one_hundred_percent_rejected(self) -> None:
        with pytest.raises(ValueError, match="spot_pct"):
            Scenario("wipeout", spot_pct=-120.0)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_fx_delta_rejects_non_finite_bump(self, market, bad) -> None:
        with pytest.raises(ValueError, match="spot|finite"):
            fx_delta(FXForward(10e6, 1.09, 1.0), market, rel_bump=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_dv01_rejects_non_finite_bump(self, market, bad) -> None:
        with pytest.raises(ValueError, match="finite"):
            dv01(FXForward(10e6, 1.09, 1.0), market, "USD", bump_bp=bad)

    @pytest.mark.parametrize("bad", NON_FINITE)
    def test_forward_carry_rejects_non_finite_horizon(self, market,
                                                      bad) -> None:
        with pytest.raises(ValueError, match="horizon"):
            forward_carry(FXForward(10e6, 1.09, 1.0), market, bad)

    def test_finite_risk_numbers_are_produced(self, market) -> None:
        fwd = FXForward(10e6, 1.09, 1.0)
        # Long 10m EUR forward: FX delta is the PV'd base notional.
        d = fx_delta(fwd, market)
        assert d == pytest.approx(
            10e6 * market.foreign_curve_adjusted.df(1.0), rel=1e-8)
        # Domestic and foreign DV01s of an outright have opposite signs.
        dv_d = dv01(fwd, market, "USD")
        dv_f = dv01(fwd, market, "EUR")
        assert np.isfinite(dv_d) and np.isfinite(dv_f)
        assert dv_d * dv_f < 0.0


class TestDayCountUnaffected:
    """The date layer takes no floats, so it needs no NaN guard — verify."""

    def test_year_fraction_conventions_still_exact(self) -> None:
        from fx_rates import year_fraction

        a, b = dt.date(2024, 1, 31), dt.date(2024, 7, 31)
        actual_days = (b - a).days
        assert year_fraction(a, b, "ACT/360") == pytest.approx(
            actual_days / 360.0, abs=1e-15)
        assert year_fraction(a, b, "ACT/365F") == pytest.approx(
            actual_days / 365.0, abs=1e-15)
        # ACT/360 exceeds ACT/365F for the same actual days (smaller divisor).
        assert year_fraction(a, b, "ACT/360") > year_fraction(a, b, "ACT/365F")
        # 30/360 US end-of-month rule: 31 Jan -> 31 Jul is exactly half a year.
        assert year_fraction(a, b, "30/360") == pytest.approx(0.5, abs=1e-15)
