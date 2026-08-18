"""Universe: cross construction, triangular consistency, screens, pip conventions."""

import numpy as np
import pandas as pd
import pytest

from fx_pairs.data import synthetic as syn
from fx_pairs.universe import (
    DEFAULT_PIP_SPREADS,
    correlation_screen,
    enumerate_candidate_pairs,
    make_cross,
    market_pair,
    market_price_from_legs,
    pip_size,
    pip_spread,
    triangular_spread,
)


@pytest.fixture(scope="module")
def legs():
    legs, _ = syn.make_two_block_panel(n=600, block_a=("AUD", "NZD", "CAD"),
                                       block_b=("JPY", "CHF"), seed=2)
    return legs


class TestCrossConstruction:
    def test_cross_is_exact_ratio_of_usd_legs(self, legs):
        cross = make_cross(legs, "AUD", "NZD")
        expected = legs["AUD"] / legs["NZD"]
        assert np.max(np.abs(cross.values - expected.values)) == 0.0

    def test_cross_vs_usd_equals_leg(self, legs):
        assert np.allclose(make_cross(legs, "AUD", "USD").values,
                           legs["AUD"].values, rtol=0, atol=0)

    def test_usd_quote_pair_is_reciprocal_of_leg(self, legs):
        usdjpy = make_cross(legs, "USD", "JPY")
        assert np.allclose(usdjpy.values, 1.0 / legs["JPY"].values, rtol=1e-15)

    def test_market_price_from_legs_matches_make_cross(self, legs):
        a = market_price_from_legs(legs, "AUDJPY")
        b = make_cross(legs, "AUD", "JPY")
        assert np.array_equal(a.values, b.values)

    def test_triangular_spread_identically_zero(self, legs):
        tri = triangular_spread(legs, "AUD", "USD", "JPY")
        # exact no-arbitrage identity up to float rounding
        assert np.max(np.abs(tri.values)) < 1e-12
        assert np.std(tri.values) < 1e-13

    def test_same_currency_cross_raises(self, legs):
        with pytest.raises(ValueError):
            make_cross(legs, "AUD", "AUD")

    def test_unknown_currency_raises(self, legs):
        with pytest.raises(KeyError):
            make_cross(legs, "AUD", "XXX")


class TestConventions:
    def test_market_pair_base_conventions(self):
        assert market_pair("EUR") == "EURUSD"
        assert market_pair("AUD") == "AUDUSD"
        assert market_pair("JPY") == "USDJPY"
        assert market_pair("MXN") == "USDMXN"

    def test_market_pair_usd_raises(self):
        with pytest.raises(ValueError):
            market_pair("USD")

    def test_pip_size_jpy_vs_other(self):
        assert pip_size("USDJPY") == 0.01
        assert pip_size("EURJPY") == 0.01
        assert pip_size("EURUSD") == 1e-4
        assert pip_size("USDMXN") == 1e-4

    def test_pip_spread_em_wider_than_majors(self):
        assert pip_spread("USDZAR") > pip_spread("USDMXN") > pip_spread("EURUSD")
        assert DEFAULT_PIP_SPREADS["EURUSD"] <= 1.0

    def test_pip_spread_override_and_default(self):
        assert pip_spread("EURUSD", {"EURUSD": 0.3}) == 0.3
        assert pip_spread("ABCXYZ") == 5.0  # unknown -> conservative wide


class TestScreens:
    def test_enumerate_candidate_pairs(self):
        pairs = enumerate_candidate_pairs(["A", "B", "C", "D"])
        assert len(pairs) == 6
        assert ("A", "B") in pairs
        assert all(a != b for a, b in pairs)
        # unordered: no duplicates in reverse orientation
        assert ("B", "A") not in pairs

    def test_correlation_screen_finds_block_pairs(self, legs):
        prices = pd.DataFrame({
            "AUDUSD": make_cross(legs, "AUD", "USD"),
            "NZDUSD": make_cross(legs, "NZD", "USD"),
            "USDJPY": make_cross(legs, "USD", "JPY"),
        })
        out = correlation_screen(prices, min_abs_corr=0.5)
        top = (out.iloc[0]["pair_1"], out.iloc[0]["pair_2"])
        assert set(top) == {"AUDUSD", "NZDUSD"}
        assert out.iloc[0]["corr"] > 0.7

    def test_correlation_screen_excludes_low_corr(self, legs):
        rng = np.random.default_rng(0)
        idx = legs.index
        indep = pd.Series(np.exp(np.cumsum(0.006 * rng.standard_normal(len(idx)))),
                          index=idx)
        prices = pd.DataFrame({"AUDUSD": make_cross(legs, "AUD", "USD"),
                               "INDEP": indep})
        out = correlation_screen(prices, min_abs_corr=0.5)
        assert len(out) == 0

    def test_pegged_pair_dropped_with_warning(self, legs):
        prices = pd.DataFrame({
            "AUDUSD": make_cross(legs, "AUD", "USD"),
            "NZDUSD": make_cross(legs, "NZD", "USD"),
            "PEGUSD": syn.make_pegged_pair(n=len(legs)).reindex(legs.index).ffill(),
        })
        prices["PEGUSD"] = 3.6725
        with pytest.warns(UserWarning, match="pegged"):
            out = correlation_screen(prices, min_abs_corr=0.0)
        assert not (out[["pair_1", "pair_2"]] == "PEGUSD").any().any()

    def test_screen_requires_two_instruments(self, legs):
        with pytest.raises(ValueError):
            correlation_screen(legs[["AUD"]])
