"""Bootstrap round-trips, order independence, variants, failure modes."""

from __future__ import annotations

import numpy as np
import pytest

import fi_rates as fr
from fi_rates.bootstrap import FRA, BondQuote, Deposit, ParSwap
from fi_rates.data import CURVE_VARIANTS, market_quotes

TOL = 1e-10


class TestRoundTrip:
    @pytest.mark.parametrize("interp", ["loglinear_df", "linear_zero"])
    def test_deposits_and_swaps_reprice_to_1e10(self, quotes, interp):
        curve = fr.bootstrap_curve(quotes, interpolation=interp)
        for ins, err in fr.reprice_instruments(quotes, curve):
            assert abs(err) < TOL, f"{ins} error {err:.2e}"

    def test_fra_repricing(self):
        instruments = [
            Deposit(0.5, 0.030),
            FRA(0.5, 1.0, 0.032),
            ParSwap(2.0, 0.033),
        ]
        curve = fr.bootstrap_curve(instruments)
        assert curve.simple_forward_rate(0.5, 1.0) == pytest.approx(0.032, abs=TOL)
        for _, err in fr.reprice_instruments(instruments, curve):
            assert abs(err) < TOL

    def test_pillars_match_instrument_maturities(self, quotes, curve):
        expected = sorted(ins.pillar for ins in quotes)
        np.testing.assert_allclose(curve.times, expected, atol=1e-12)

    def test_semiannual_swap_bootstrap(self):
        instruments = [Deposit(0.5, 0.03), ParSwap(1.0, 0.031, 2), ParSwap(2.0, 0.033, 2)]
        curve = fr.bootstrap_curve(instruments)
        assert curve.par_rate(2.0, 2) == pytest.approx(0.033, abs=TOL)


class TestOrderIndependence:
    def test_shuffled_inputs_same_curve(self, quotes):
        rng = np.random.default_rng(7)
        shuffled = list(quotes)
        rng.shuffle(shuffled)
        c1 = fr.bootstrap_curve(quotes)
        c2 = fr.bootstrap_curve(shuffled)
        np.testing.assert_allclose(c1.dfs, c2.dfs, rtol=0, atol=1e-15)


class TestVariants:
    @pytest.mark.parametrize("variant", CURVE_VARIANTS)
    def test_all_variants_bootstrap_and_reprice(self, variant):
        qs = market_quotes(variant, seed=1)
        curve = fr.bootstrap_curve(qs)
        for _, err in fr.reprice_instruments(qs, curve):
            assert abs(err) < TOL

    def test_inverted_curve_zero_slope(self):
        curve = fr.bootstrap_curve(market_quotes("inverted", noise_bp=0.0))
        zeros = curve.zero_rates
        assert zeros[0] > zeros[-1]  # short above long

    def test_negative_rate_curve_dfs_above_one(self):
        curve = fr.bootstrap_curve(market_quotes("negative", noise_bp=0.0))
        assert float(np.asarray(curve.df(1.0))) > 1.0
        assert curve.zero_rates[-1] > 0  # long end mildly positive

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="variant"):
            market_quotes("humped")


class TestBondBootstrap:
    def test_round_trip_bond_prices(self):
        # build quotes from a known curve, bootstrap back, compare
        base = fr.DiscountCurve.from_zero_rates(
            [1.0, 2.0, 3.0, 5.0], [0.03, 0.033, 0.035, 0.038]
        )
        quotes = []
        for mat, cpn in [(1.0, 0.03), (2.0, 0.035), (3.0, 0.04), (5.0, 0.042)]:
            q = BondQuote(maturity=mat, coupon=cpn, dirty_price=0.0, frequency=2)
            t, cf = q.cashflows()
            price = float(np.sum(cf * np.asarray(base.df(t))))
            quotes.append(BondQuote(mat, cpn, price, 2))
        curve = fr.bootstrap_bond_curve(quotes)
        np.testing.assert_allclose(
            curve.dfs, np.asarray(base.df(np.array([1.0, 2.0, 3.0, 5.0]))), atol=TOL
        )

    def test_zero_coupon_quote_recovers_df(self):
        quotes = [BondQuote(maturity=2.0, coupon=0.0, dirty_price=0.92, frequency=1)]
        curve = fr.bootstrap_bond_curve(quotes)
        assert float(np.asarray(curve.df(2.0))) == pytest.approx(0.92, abs=TOL)

    def test_empty_quotes_raise(self):
        with pytest.raises(ValueError, match="no bond quotes"):
            fr.bootstrap_bond_curve([])

    def test_duplicate_maturity_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            fr.bootstrap_bond_curve(
                [BondQuote(2.0, 0.03, 0.95), BondQuote(2.0, 0.04, 0.97)]
            )

    def test_unsolvable_bond_price_raises_informative(self):
        # price 10x above any attainable PV -> no admissible discount factor
        with pytest.raises(ValueError, match="bond bootstrap failed"):
            fr.bootstrap_bond_curve([BondQuote(2.0, 0.03, 50.0, 1)])


class TestFailureModes:
    def test_empty_instruments_raise(self):
        with pytest.raises(ValueError, match="no instruments"):
            fr.bootstrap_curve([])

    def test_duplicate_pillar_raises(self):
        with pytest.raises(ValueError, match="duplicate pillar"):
            fr.bootstrap_curve([Deposit(1.0, 0.03), ParSwap(1.0, 0.031)])

    def test_unsolvable_deposit_raises_informative(self):
        # rate < -1/T has no positive discount factor
        with pytest.raises(ValueError, match="bootstrap failed at pillar"):
            fr.bootstrap_curve([Deposit(1.0, -2.0)])

    def test_error_message_names_instrument(self):
        with pytest.raises(ValueError, match="Deposit"):
            fr.bootstrap_curve([Deposit(1.0, -2.0)])

    def test_negative_deposit_maturity_raises(self):
        with pytest.raises(ValueError, match="maturity"):
            fr.bootstrap_curve([Deposit(-1.0, 0.03)])

    def test_invalid_fra_raises(self):
        with pytest.raises(ValueError, match="FRA"):
            fr.bootstrap_curve([FRA(1.0, 0.5, 0.03)])

    def test_invalid_swap_frequency_raises(self):
        with pytest.raises(ValueError, match="frequency"):
            fr.bootstrap_curve([ParSwap(2.0, 0.03, frequency=3)])

    def test_fractional_swap_maturity_raises(self):
        with pytest.raises(ValueError, match="whole number"):
            fr.bootstrap_curve([ParSwap(1.5, 0.03, frequency=1)])
