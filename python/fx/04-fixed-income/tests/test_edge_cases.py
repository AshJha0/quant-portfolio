"""Edge cases from the documentation contract: negative rates, zero
notional, same-currency crosses, settlement edges, empty books, and the
basis-mispricing demonstration."""

import datetime as dt

import numpy as np
import pytest

from fx_rates.curve import DiscountCurve
from fx_rates.daycount import spot_date, year_fraction
from fx_rates.fxforward import (
    FXForward,
    FXSwap,
    MarketState,
    cip_forward,
    forward_points,
    market_forward,
)
from fx_rates.risk import basis_dv01, book_risk_report, dv01, fx_delta
from fx_rates.data.live import network_allowed


class TestNegativeRatesEra:
    """EUR 2019: r_f < r_d with r_f < 0 — economic sign checks."""

    def test_forward_points_positive_when_foreign_rate_below_domestic(
        self, negative_eur_market
    ):
        m = negative_eur_market
        assert m.foreign_curve.zero_rate(1.0) < 0.0 < m.domestic_curve.zero_rate(1.0)
        for t in [0.25, 1.0, 5.0]:
            assert forward_points(m, t) > 0.0

    def test_cip_holds_with_negative_rates(self, negative_eur_market):
        m = negative_eur_market
        for t in [0.5, 2.0, 7.0]:
            f = cip_forward(m.spot, m.domestic_curve, m.foreign_curve, t)
            assert f / m.spot == pytest.approx(
                m.foreign_curve.df(t) / m.domestic_curve.df(t), abs=1e-14
            )

    def test_forward_mtm_methods_agree_with_negative_rates(self, negative_eur_market):
        m = negative_eur_market
        fwd = FXForward(10e6, market_forward(m, 3.0) * 1.01, 3.0)
        assert fwd.value(m, "cashflows") == pytest.approx(
            fwd.value(m, "forward"), abs=1e-8
        )


class TestZeroNotional:
    def test_zero_notional_forward_valid_and_riskless(self, market):
        fwd = FXForward(0.0, 1.10, 1.0)
        assert fwd.value(market) == 0.0
        assert fx_delta(fwd, market) == pytest.approx(0.0, abs=1e-12)
        assert dv01(fwd, market, "USD") == pytest.approx(0.0, abs=1e-12)
        assert basis_dv01(fwd, market) == pytest.approx(0.0, abs=1e-12)

    def test_zero_notional_fx_swap(self, market):
        swap = FXSwap(0.0, 1.09, 0.25, 1.11, 1.0)
        assert swap.value(market) == 0.0


class TestSameCurrencyCross:
    def test_forward_same_ccy_rejected(self):
        with pytest.raises(ValueError, match="rejected"):
            FXForward(1e6, 1.0, 1.0, pair=("USD", "USD"))

    def test_swap_same_ccy_rejected(self):
        with pytest.raises(ValueError, match="rejected"):
            FXSwap(1e6, 1.0, 0.5, 1.0, 1.0, pair=("EUR", "EUR"))


class TestInvalidInstruments:
    @pytest.mark.parametrize("kwargs", [
        dict(notional_base=1e6, strike=-1.1, expiry=1.0),
        dict(notional_base=1e6, strike=1.1, expiry=0.0),
        dict(notional_base=1e6, strike=1.1, expiry=-1.0),
    ])
    def test_bad_forward_parameters_raise(self, kwargs):
        with pytest.raises(ValueError):
            FXForward(**kwargs)


class TestSettlementEdges:
    def test_spot_date_over_month_end(self):
        assert spot_date(dt.date(2024, 1, 30)) == dt.date(2024, 2, 1)

    def test_spot_date_over_year_end(self):
        assert spot_date(dt.date(2024, 12, 31)) == dt.date(2025, 1, 2)

    def test_one_day_accrual(self):
        d = dt.date(2024, 6, 14)
        assert year_fraction(d, d + dt.timedelta(days=1), "ACT/360") == pytest.approx(
            1 / 360, abs=1e-15
        )

    def test_very_short_dated_forward_prices(self, market):
        fwd = FXForward(1e6, market_forward(market, 1 / 365), 1 / 365)
        assert fwd.value(market) == pytest.approx(0.0, abs=1e-6)


class TestEmptyBook:
    def test_empty_book_report(self, market):
        rep = book_risk_report([], market)
        assert rep.loc["TOTAL", "pv"] == 0.0
        assert rep.shape[0] == 1


class TestBasisMispricing:
    """Ignoring the basis misprices a 5y EURUSD forward (README numbers)."""

    def test_5y_mispricing_exceeds_100_pips_in_normal_regime(self, market):
        f_cip = cip_forward(market.spot, market.domestic_curve, market.foreign_curve, 5.0)
        f_mkt = market_forward(market, 5.0)
        assert (f_mkt - f_cip) * 1e4 > 100.0  # > 100 pips at -25bp basis

    def test_pv_error_on_100m_5y_forward(self, market):
        # value a 5y forward struck at the *market* forward but priced by a
        # CIP-only model: the model shows a fictitious positive PV
        k = market_forward(market, 5.0)
        fwd = FXForward(100e6, k, 5.0)
        cip_model = market.replace(basis_spreads=())
        pv_true = fwd.value(market)
        pv_cip_model = fwd.value(cip_model)
        assert pv_true == pytest.approx(0.0, abs=1e-4)
        assert pv_cip_model < -1e6  # CIP model misprices by over $1m

    def test_crisis_regime_mispricing_is_larger(self, crisis_market, market):
        def mispricing(m):
            return abs(
                market_forward(m, 1.0)
                - cip_forward(m.spot, m.domestic_curve, m.foreign_curve, 1.0)
            ) / m.spot
        assert mispricing(crisis_market) > 3.0 * mispricing(market)


class TestOffline:
    def test_network_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("FX_RATES_ALLOW_NETWORK", raising=False)
        assert not network_allowed()
        from fx_rates.data.live import load_frankfurter_spot
        with pytest.raises(RuntimeError, match="Network access is disabled"):
            load_frankfurter_spot()

    def test_extreme_but_valid_curve_inputs(self):
        # 50% rates and -5% rates both produce valid curves
        hi = DiscountCurve.from_zero_rates([1.0, 5.0], [0.50, 0.45])
        lo = DiscountCurve.from_zero_rates([1.0, 5.0], [-0.05, -0.04])
        assert 0 < hi.df(5.0) < 1 < lo.df(5.0)
        m = MarketState(1.0, hi, lo, pair=("XXX", "YYY"))
        assert market_forward(m, 5.0) > m.spot
