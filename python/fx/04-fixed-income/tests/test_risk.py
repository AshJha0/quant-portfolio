"""Risk measures: FX delta formula, DV01 economic signs, KRD locality,
basis DV01, book aggregation."""

import numpy as np
import pytest

from fx_rates.fxforward import FXForward, FXSwap, market_forward
from fx_rates.risk import (
    basis_dv01,
    book_risk_report,
    book_value,
    dv01,
    fx_delta,
    key_rate_dv01,
    position_risk,
)


@pytest.fixture
def long_fwd(market):
    return FXForward(10e6, market_forward(market, 2.0), 2.0)


class TestFXDelta:
    def test_unit_forward_delta_equals_foreign_df_times_notional(self, market, long_fwd):
        # V = N(S*DF_f_adj - K*DF_d) => dV/dS = N * DF_f_adj(T), exactly
        expected = 10e6 * market.foreign_curve_adjusted.df(2.0)
        assert fx_delta(long_fwd, market) == pytest.approx(expected, rel=1e-9)

    def test_short_forward_delta_negative(self, market):
        short = FXForward(-10e6, 1.15, 2.0)
        assert fx_delta(short, market) < 0.0

    def test_delta_times_move_predicts_linear_pnl(self, market, long_fwd):
        d = fx_delta(long_fwd, market)
        ds = 0.02
        moved = market.replace(spot=market.spot + ds)
        assert long_fwd.value(moved) - long_fwd.value(market) == pytest.approx(
            d * ds, rel=1e-9
        )

    def test_zero_notional_zero_delta(self, market):
        assert fx_delta(FXForward(0.0, 1.1, 1.0), market) == pytest.approx(0.0, abs=1e-9)


class TestDV01Signs:
    def test_long_forward_domestic_dv01_positive(self, market, long_fwd):
        # USD rates up => pay-USD leg discounts harder => long base gains
        assert dv01(long_fwd, market, "USD") > 0.0

    def test_long_forward_foreign_dv01_negative_sign_flip(self, market, long_fwd):
        # EUR rates up => receive-EUR leg worth less: opposite sign to USD
        d_usd = dv01(long_fwd, market, "USD")
        d_eur = dv01(long_fwd, market, "EUR")
        assert d_eur < 0.0 < d_usd

    def test_short_forward_flips_both_signs(self, market):
        short = FXForward(-10e6, market_forward(market, 2.0), 2.0)
        assert dv01(short, market, "USD") < 0.0
        assert dv01(short, market, "EUR") > 0.0

    def test_dv01_magnitude_scales_with_maturity(self, market):
        f1 = FXForward(10e6, market_forward(market, 1.0), 1.0)
        f5 = FXForward(10e6, market_forward(market, 5.0), 5.0)
        assert abs(dv01(f5, market, "USD")) > 3.0 * abs(dv01(f1, market, "USD"))

    def test_dv01_hand_check_against_analytic(self, market, long_fwd):
        # analytic: dV/dz_d = N*K*T*DF_d(T) per unit rate => per bp * 1e-4
        n, k, t = 10e6, long_fwd.strike, 2.0
        analytic = n * k * t * market.domestic_curve.df(t) * 1e-4
        assert dv01(long_fwd, market, "USD") == pytest.approx(analytic, rel=1e-6)

    def test_unknown_currency_raises(self, market, long_fwd):
        with pytest.raises(ValueError, match="unknown currency"):
            dv01(long_fwd, market, "GBP")


class TestKeyRateDV01:
    def test_krd_locality(self, market, long_fwd):
        # a single 2y cashflow has KRD only at the 2y pillar (2.0 is a pillar)
        krd = key_rate_dv01(long_fwd, market, "USD")
        assert abs(krd[2.0]) > 0.0
        for t, v in krd.items():
            if t != 2.0:
                assert v == pytest.approx(0.0, abs=1e-6)

    def test_krd_sums_to_parallel_dv01(self, market, long_fwd):
        krd = key_rate_dv01(long_fwd, market, "USD")
        assert krd.sum() == pytest.approx(dv01(long_fwd, market, "USD"), rel=1e-6)

    def test_foreign_krd_sums_to_foreign_dv01(self, market, long_fwd):
        krd = key_rate_dv01(long_fwd, market, "EUR")
        assert krd.sum() == pytest.approx(dv01(long_fwd, market, "EUR"), rel=1e-6)

    def test_off_pillar_cashflow_hits_adjacent_buckets_only(self, market):
        fwd = FXForward(10e6, 1.15, 2.5)  # between the 2y and 3y pillars
        krd = key_rate_dv01(fwd, market, "USD")
        hit = {t for t, v in krd.items() if abs(v) > 1e-6}
        assert hit == {2.0, 3.0}


class TestBasisDV01:
    def test_long_forward_basis_dv01_negative(self, market, long_fwd):
        # basis less negative (+1bp) => adjusted EUR DFs fall => long base loses
        assert basis_dv01(long_fwd, market) < 0.0

    def test_basis_dv01_zero_for_domestic_only_exposure(self, market):
        # a position with offsetting base-ccy cashflows at the same date has
        # no basis risk; approximate with zero-notional forward
        assert basis_dv01(FXForward(0.0, 1.1, 1.0), market) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_basis_dv01_matches_foreign_zero_shift(self, market, long_fwd):
        # shifting the basis by 1bp is economically identical to shifting the
        # adjusted foreign zeros by 1bp => equals foreign-curve DV01
        assert basis_dv01(long_fwd, market) == pytest.approx(
            dv01(long_fwd, market, "EUR"), rel=1e-9
        )


class TestBookAggregation:
    def test_report_totals_are_sums(self, market, book):
        rep = book_risk_report(book, market)
        body = rep.drop(index="TOTAL")
        assert np.allclose(rep.loc["TOTAL"].values, body.sum(axis=0).values, atol=1e-6)

    def test_report_pv_matches_book_value(self, market, book):
        rep = book_risk_report(book, market)
        assert rep.loc["TOTAL", "pv"] == pytest.approx(book_value(book, market), abs=1e-6)

    def test_empty_book_zero_total(self, market):
        rep = book_risk_report([], market)
        assert list(rep.index) == ["TOTAL"]
        assert np.allclose(rep.loc["TOTAL"].values.astype(float), 0.0)
        assert book_value([], market) == 0.0

    def test_position_risk_keys(self, market, long_fwd):
        r = position_risk(long_fwd, market)
        assert set(r) == {"pv", "fx_delta", "dv01_usd", "dv01_eur", "basis_dv01"}

    def test_fx_swap_in_book_has_small_delta_but_real_dv01(self, market):
        swap = FXSwap(40e6, market_forward(market, 0.25), 0.25,
                      market_forward(market, 1.0), 1.0)
        outright_delta = 40e6 * market.foreign_curve_adjusted.df(1.0)
        assert abs(fx_delta(swap, market)) < 0.05 * outright_delta
        assert abs(dv01(swap, market, "USD")) > 0.0
