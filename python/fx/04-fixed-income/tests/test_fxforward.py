"""CIP identities, triangular consistency, forward MTM and FX swaps."""

import numpy as np
import pytest

from fx_rates.data import generate_market_quotes, build_market_state, third_currency_curve
from fx_rates.fxforward import (
    FXForward,
    FXSwap,
    MarketState,
    cip_forward,
    forward_points,
    forward_points_table,
    market_forward,
)


class TestCIPIdentities:
    def test_f_over_s_equals_df_ratio_exactly(self, market):
        for t in [0.1, 0.25, 1.0, 3.7, 5.0, 10.0]:
            f = cip_forward(market.spot, market.domestic_curve, market.foreign_curve, t)
            lhs = f / market.spot
            rhs = market.foreign_curve.df(t) / market.domestic_curve.df(t)
            assert lhs == pytest.approx(rhs, abs=1e-14)

    def test_forward_at_t0_is_spot(self, market):
        assert cip_forward(
            market.spot, market.domestic_curve, market.foreign_curve, 0.0
        ) == pytest.approx(market.spot, abs=1e-15)

    def test_market_forward_embeds_basis(self, market):
        # negative basis => market forward above CIP forward
        f_cip = cip_forward(market.spot, market.domestic_curve, market.foreign_curve, 5.0)
        f_mkt = market_forward(market, 5.0)
        assert f_mkt > f_cip

    def test_zero_basis_market_forward_equals_cip(self, market):
        m0 = market.replace(basis_spreads=())
        for t in [0.25, 1.0, 5.0]:
            assert market_forward(m0, t) == pytest.approx(
                cip_forward(m0.spot, m0.domestic_curve, m0.foreign_curve, t),
                abs=1e-14,
            )

    def test_triangular_consistency_to_1e12(self, market):
        # EURJPY forward must equal EURUSD forward * USDJPY forward when
        # spots are triangular-consistent — with the same three curves this
        # is an exact identity of CIP.
        usd = market.domestic_curve
        eur = market.foreign_curve
        jpy = third_currency_curve(seed=7)
        s_eurusd = market.spot
        s_usdjpy = 148.50
        s_eurjpy = s_eurusd * s_usdjpy
        for t in [0.25, 1.0, 2.0, 5.0, 10.0]:
            f_eurusd = cip_forward(s_eurusd, usd, eur, t)   # dom=USD, for=EUR
            f_usdjpy = cip_forward(s_usdjpy, jpy, usd, t)   # dom=JPY, for=USD
            f_eurjpy = cip_forward(s_eurjpy, jpy, eur, t)   # dom=JPY, for=EUR
            assert f_eurjpy == pytest.approx(f_eurusd * f_usdjpy, rel=1e-12)

    def test_points_sign_follows_rate_differential(self, market):
        # EUR rates < USD rates => EURUSD forward premium: positive points
        assert market.foreign_curve.zero_rate(1.0) < market.domestic_curve.zero_rate(1.0)
        assert forward_points(market, 1.0) > 0.0

    def test_points_table_columns_and_consistency(self, market):
        tbl = forward_points_table(market, [0.25, 1.0, 5.0])
        assert list(tbl["tenor_y"]) == [0.25, 1.0, 5.0]
        # market points - cip points == basis effect
        assert np.allclose(
            tbl["market_points"] - tbl["cip_points"],
            tbl["basis_points_effect"], atol=1e-9,
        )
        # basis effect negative-basis => positive points effect
        assert (tbl["basis_points_effect"] > 0).all()
        assert (tbl["basis_spread_bp"] < 0).all()

    def test_bad_spot_raises(self, market):
        with pytest.raises(ValueError, match="spot"):
            cip_forward(0.0, market.domestic_curve, market.foreign_curve, 1.0)


class TestFXForwardMTM:
    def test_at_inception_value_zero(self, market):
        k = market_forward(market, 2.0)
        fwd = FXForward(10e6, k, 2.0)
        assert fwd.value(market) == pytest.approx(0.0, abs=1e-6)  # 1e-13 relative

    def test_two_valuation_methods_agree_after_moves(self, market):
        fwd = FXForward(10e6, market_forward(market, 2.0), 2.0)
        moved = market.replace(
            spot=market.spot * 1.03,
            domestic_curve=market.domestic_curve.parallel_shift(35.0),
            foreign_curve=market.foreign_curve.parallel_shift(-20.0),
        )
        v_cf = fwd.value(moved, method="cashflows")
        v_fw = fwd.value(moved, method="forward")
        assert v_cf != 0.0
        assert v_fw == pytest.approx(v_cf, abs=1e-10 * abs(v_cf) + 1e-8)

    def test_mtm_sign_after_spot_rally(self, market):
        fwd = FXForward(10e6, market_forward(market, 1.0), 1.0)
        up = market.replace(spot=market.spot * 1.05)
        assert fwd.value(up) > 0.0  # long base gains when base appreciates

    def test_hand_check_cashflow_valuation(self, market):
        n, k, t = 5e6, 1.20, 3.0
        fwd = FXForward(n, k, t)
        expected = (
            n * market.spot * market.foreign_curve_adjusted.df(t)
            - n * k * market.domestic_curve.df(t)
        )
        assert fwd.value(market) == pytest.approx(expected, abs=1e-9)

    def test_short_position_is_negative_of_long(self, market):
        long = FXForward(10e6, 1.15, 2.0)
        short = FXForward(-10e6, 1.15, 2.0)
        assert short.value(market) == pytest.approx(-long.value(market), abs=1e-9)

    def test_cashflows_enumeration(self):
        fwd = FXForward(10e6, 1.10, 1.0)
        cfs = fwd.cashflows()
        assert ("EUR", 1.0, 10e6) in cfs
        assert ("USD", 1.0, -11e6) in cfs

    def test_unknown_method_raises(self, market):
        with pytest.raises(ValueError, match="method"):
            FXForward(1e6, 1.1, 1.0).value(market, method="magic")

    def test_pair_mismatch_raises(self, market):
        fwd = FXForward(1e6, 150.0, 1.0, pair=("USD", "JPY"))
        with pytest.raises(ValueError, match="pair"):
            fwd.value(market)


class TestFXSwap:
    def test_equals_sum_of_two_forwards_identity(self, market):
        swap = FXSwap(40e6, market_forward(market, 0.25), 0.25,
                      market_forward(market, 1.0) * 1.001, 1.0)
        near, far = swap.legs()
        direct = swap.value(market)
        decomposed = near.value(market) + far.value(market)
        assert direct == pytest.approx(decomposed, abs=1e-8)

    def test_identity_holds_after_market_moves(self, market):
        swap = FXSwap(40e6, 1.09, 0.25, 1.11, 1.0)
        moved = market.replace(
            spot=market.spot * 0.97,
            domestic_curve=market.domestic_curve.parallel_shift(-50.0),
        )
        near, far = swap.legs()
        assert swap.value(moved) == pytest.approx(
            near.value(moved) + far.value(moved), abs=1e-8
        )

    def test_at_market_strikes_swap_pv_zero(self, market):
        swap = FXSwap(40e6, market_forward(market, 0.25), 0.25,
                      market_forward(market, 1.0), 1.0)
        assert swap.value(market) == pytest.approx(0.0, abs=1e-6)

    def test_spot_risk_nearly_cancels(self, market):
        # matched-notional FX swap has tiny net spot delta (DF_f(T1)-DF_f(T2))
        swap = FXSwap(40e6, market_forward(market, 0.25), 0.25,
                      market_forward(market, 1.0), 1.0)
        outright = FXForward(40e6, market_forward(market, 1.0), 1.0)
        up = market.replace(spot=market.spot * 1.01)
        assert abs(swap.value(up) - swap.value(market)) < 0.05 * abs(
            outright.value(up) - outright.value(market)
        )

    def test_bad_leg_order_raises(self):
        with pytest.raises(ValueError, match="near_expiry"):
            FXSwap(1e6, 1.1, 1.0, 1.12, 0.5)


class TestMarketState:
    def test_same_currency_pair_rejected(self, market):
        with pytest.raises(ValueError, match="rejected"):
            MarketState(1.0, market.domestic_curve, market.foreign_curve,
                        pair=("USD", "USD"))

    def test_negative_spot_rejected(self, market):
        with pytest.raises(ValueError, match="spot"):
            MarketState(-1.1, market.domestic_curve, market.foreign_curve)

    def test_replace_rebuilds_adjusted_curve(self, market):
        _ = market.foreign_curve_adjusted  # populate cache
        m2 = market.replace(basis_spreads=((5.0, -0.01),))
        assert m2.foreign_curve_adjusted.df(5.0) != pytest.approx(
            market.foreign_curve_adjusted.df(5.0), abs=1e-9
        )
