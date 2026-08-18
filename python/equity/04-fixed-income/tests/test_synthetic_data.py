"""Synthetic market generators: determinism, shape, portfolio properties."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

import fi_rates as fr
from fi_rates.bootstrap import Deposit, ParSwap
from fi_rates.data import CURVE_VARIANTS, market_quotes, sample_portfolio

D = dt.date


class TestMarketQuotes:
    def test_deterministic_given_seed(self):
        a = market_quotes("upward", seed=123)
        b = market_quotes("upward", seed=123)
        assert [i.rate for i in a] == [i.rate for i in b]

    def test_different_seed_different_noise(self):
        a = market_quotes("upward", seed=1)
        b = market_quotes("upward", seed=2)
        assert [i.rate for i in a] != [i.rate for i in b]

    def test_structure_deposits_then_swaps(self):
        qs = market_quotes("upward")
        assert all(isinstance(q, Deposit) for q in qs[:3])
        assert all(isinstance(q, ParSwap) for q in qs[3:])
        pillars = [q.pillar for q in qs]
        assert pillars == sorted(pillars)

    def test_upward_variant_is_upward(self):
        qs = market_quotes("upward", noise_bp=0.0)
        rates = [q.rate for q in qs]
        assert all(a < b for a, b in zip(rates, rates[1:]))

    def test_inverted_variant_is_inverted(self):
        qs = market_quotes("inverted", noise_bp=0.0)
        assert qs[0].rate > qs[-1].rate

    def test_flat_variant_is_flat(self):
        qs = market_quotes("flat", noise_bp=0.0)
        rates = [q.rate for q in qs]
        assert max(rates) - min(rates) < 1e-12

    def test_negative_variant_has_negative_short_end(self):
        qs = market_quotes("negative", noise_bp=0.0)
        assert qs[0].rate < 0.0

    def test_zero_noise_smooth(self):
        qs = market_quotes("upward", seed=1, noise_bp=0.0)
        qs2 = market_quotes("upward", seed=99, noise_bp=0.0)
        assert [q.rate for q in qs] == [q.rate for q in qs2]

    @pytest.mark.parametrize("variant", CURVE_VARIANTS)
    def test_all_variants_have_11_quotes(self, variant):
        assert len(market_quotes(variant)) == 11


class TestSamplePortfolio:
    def test_deterministic(self, settlement):
        a = sample_portfolio(settlement, seed=42)
        b = sample_portfolio(settlement, seed=42)
        assert [(p.label, p.quantity, p.z_spread) for p in a] == [
            (p.label, p.quantity, p.z_spread) for p in b
        ]

    def test_govt_zero_corp_positive_spread(self, settlement):
        for pos in sample_portfolio(settlement):
            if pos.label.startswith("GOVT"):
                assert pos.z_spread == 0.0
            else:
                assert 0.008 <= pos.z_spread <= 0.025

    def test_all_bonds_alive_at_settlement(self, settlement):
        for pos in sample_portfolio(settlement):
            assert pos.bond.maturity > settlement
            assert pos.bond.effective < settlement

    def test_portfolio_prices_on_bootstrapped_curve(self, settlement, curve):
        mv = fr.portfolio_value(sample_portfolio(settlement), settlement, curve)
        assert 1.5e6 < mv < 2.0e6  # seven positions, ~par prices


class TestLiveLoaderGuard:
    def test_live_module_not_imported_by_package(self):
        import sys

        import fi_rates  # noqa: F401
        import fi_rates.data  # noqa: F401

        assert "fi_rates.data.live" not in sys.modules

    def test_live_loader_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        from fi_rates.data.live import load_fred_cmt

        with pytest.raises(RuntimeError, match="API key"):
            load_fred_cmt()  # refuses before any network access
