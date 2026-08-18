"""Synthetic data generators: determinism, regime properties, book sanity."""

import numpy as np
import pytest

from fx_rates.data import (
    REGIMES,
    build_market_state,
    generate_market_quotes,
    sample_book,
    third_currency_curve,
)
from fx_rates.fxforward import FXForward, FXSwap
from fx_rates.xccy import CrossCurrencySwap


class TestDeterminism:
    def test_same_seed_same_quotes(self):
        a = generate_market_quotes("normal", seed=123)
        b = generate_market_quotes("normal", seed=123)
        assert a.spot == b.spot
        assert a.domestic_swaps == b.domestic_swaps
        assert a.fx_forward_quotes == b.fx_forward_quotes

    def test_different_seed_different_quotes(self):
        a = generate_market_quotes("normal", seed=1)
        b = generate_market_quotes("normal", seed=2)
        assert a.spot != b.spot

    def test_unknown_regime_raises(self):
        with pytest.raises(ValueError, match="regime"):
            generate_market_quotes("hyperinflation", seed=0)


class TestRegimeProperties:
    def test_normal_regime_usd_above_eur(self):
        m = build_market_state(generate_market_quotes("normal", seed=0))
        for t in [0.5, 2.0, 5.0, 10.0]:
            assert m.domestic_curve.zero_rate(t) > m.foreign_curve.zero_rate(t)

    def test_inverted_regime_usd_curve_downward(self):
        m = build_market_state(generate_market_quotes("inverted", seed=0))
        assert m.domestic_curve.zero_rate(0.25) > m.domestic_curve.zero_rate(10.0)

    def test_negative_eur_regime_has_negative_front_end(self, negative_eur_market):
        assert negative_eur_market.foreign_curve.zero_rate(0.25) < 0.0
        # DF above one at the front end — the curve machinery supports it
        assert negative_eur_market.foreign_curve.df(0.25) > 1.0

    def test_crisis_regime_basis_wider_than_150bp(self, crisis_market):
        front = dict(crisis_market.basis_spreads)[0.25]
        assert front <= -0.0150

    @pytest.mark.parametrize("regime", REGIMES)
    def test_all_regimes_have_negative_basis(self, regime):
        q = generate_market_quotes(regime, seed=0)
        assert all(s < 0 for _, s in q.basis_spreads)

    def test_third_currency_curve_is_low_rate(self):
        jpy = third_currency_curve(seed=0)
        assert -0.01 < jpy.zero_rate(1.0) < 0.02


class TestSampleBook:
    def test_book_composition(self, market):
        book = sample_book(market, seed=1)
        kinds = [type(p) for p in book]
        assert kinds.count(FXForward) == 2
        assert kinds.count(FXSwap) == 1
        assert kinds.count(CrossCurrencySwap) == 1

    def test_book_deterministic(self, market):
        b1 = sample_book(market, seed=1)
        b2 = sample_book(market, seed=1)
        assert [p.value(market) for p in b1] == [p.value(market) for p in b2]

    def test_book_positions_have_labels_and_pv(self, market, book):
        for p in book:
            assert p.label
            assert np.isfinite(p.value(market))
