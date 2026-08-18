"""Pair candidate generation, correlation screen, SSD screen."""

import numpy as np
import pandas as pd
import pytest

from eq_pairs.data import business_index, correlated_random_walks
from eq_pairs.universe import (
    candidate_pairs,
    correlation_screen,
    log_returns,
    pair_correlations,
    ssd_distances,
    ssd_screen,
)


def _panel(cols: dict, n=None) -> pd.DataFrame:
    n = n or len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=business_index(n))


class TestCandidatePairs:
    def test_same_sector_only(self):
        sectors = {"A": "TECH", "B": "TECH", "C": "ENERGY", "D": "ENERGY"}
        pairs = candidate_pairs(["A", "B", "C", "D"], sectors)
        assert pairs == [("A", "B"), ("C", "D")]

    def test_cross_sector(self):
        pairs = candidate_pairs(["A", "B", "C"], same_sector_only=False)
        assert pairs == [("A", "B"), ("A", "C"), ("B", "C")]

    def test_requires_sectors(self):
        with pytest.raises(ValueError, match="sectors mapping required"):
            candidate_pairs(["A", "B"], same_sector_only=True)

    def test_duplicate_tickers_raise(self):
        with pytest.raises(ValueError, match="unique"):
            candidate_pairs(["A", "A"], same_sector_only=False)

    def test_missing_sector_tag_raises(self):
        with pytest.raises(ValueError, match="missing sector"):
            candidate_pairs(["A", "B"], sectors={"A": "TECH"})


class TestCorrelation:
    def test_identical_returns_corr_one(self):
        rng = np.random.default_rng(0)
        base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        prices = _panel({"A": base, "B": 2.0 * base})
        corr = pair_correlations(prices, [("A", "B")])
        assert corr.loc[[("A", "B")], "corr"].iloc[0] == pytest.approx(1.0, abs=1e-12)

    def test_price_corr_spurious_return_corr_not(self):
        """Two INDEPENDENT random walks: price-level correlation is
        spuriously high while return correlation is ~0 — the reason the
        screen runs on returns (Granger-Newbold spurious correlation)."""
        price_corrs, ret_corrs = [], []
        for seed in range(8):
            df, _ = correlated_random_walks(n=1200, rho=0.0, seed=seed)
            price_corrs.append(
                abs(pair_correlations(df, [("A", "B")], on="prices").iloc[0, 0])
            )
            ret_corrs.append(
                abs(pair_correlations(df, [("A", "B")], on="returns").iloc[0, 0])
            )
        assert np.mean(price_corrs) > 0.45  # spuriously large
        assert np.mean(ret_corrs) < 0.10  # honest: near zero
        assert np.mean(price_corrs) > 4 * np.mean(ret_corrs)

    def test_correlated_increments_pass_returns_screen(self):
        df, _ = correlated_random_walks(n=1200, rho=0.92, seed=3)
        surv = correlation_screen(df, [("A", "B")], min_corr=0.6)
        assert len(surv) == 1

    def test_screen_threshold(self):
        rng = np.random.default_rng(1)
        base = np.cumsum(rng.normal(0, 0.01, 400))
        a = 100 * np.exp(base)
        b = 100 * np.exp(base + rng.normal(0, 0.02, 400))  # decorrelated
        prices = _panel({"A": a, "B": b, "C": a * 1.01})
        surv = correlation_screen(prices, [("A", "B"), ("A", "C")], min_corr=0.95)
        assert ("A", "C") in surv.index
        assert ("A", "B") not in surv.index

    def test_zero_variance_leg_rejected_not_crashed(self):
        prices = _panel({"A": np.linspace(100, 110, 50), "FLAT": np.full(50, 42.0)})
        corr = pair_correlations(prices, [("A", "FLAT")])
        assert np.isnan(corr.iloc[0, 0])
        surv = correlation_screen(prices, [("A", "FLAT")], min_corr=-1.0)
        assert len(surv) == 0

    def test_invalid_min_corr_raises(self):
        prices = _panel({"A": np.linspace(100, 110, 50), "B": np.linspace(90, 80, 50)})
        with pytest.raises(ValueError, match="min_corr"):
            correlation_screen(prices, [("A", "B")], min_corr=1.5)

    def test_invalid_on_raises(self):
        prices = _panel({"A": np.linspace(100, 110, 50), "B": np.linspace(90, 80, 50)})
        with pytest.raises(ValueError, match="on must be"):
            pair_correlations(prices, [("A", "B")], on="levels")

    def test_log_returns_nonpositive_raises(self):
        prices = _panel({"A": np.array([100.0, -1.0, 100.0])})
        with pytest.raises(ValueError, match="positive"):
            log_returns(prices)


class TestSSD:
    def test_identical_paths_zero(self):
        rng = np.random.default_rng(2)
        base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        prices = _panel({"A": base, "B": 3.0 * base})  # same shape, scaled
        ssd = ssd_distances(prices, [("A", "B")])
        assert ssd.iloc[0, 0] == pytest.approx(0.0, abs=1e-20)

    def test_hand_computed_value(self):
        # normalised: A = [1, 1.1, 1.2], B = [1, 1.0, 1.0]
        prices = _panel({"A": np.array([10.0, 11.0, 12.0]), "B": np.array([20.0, 20.0, 20.0])})
        ssd = ssd_distances(prices, [("A", "B")])
        expected = 0.0**2 + 0.1**2 + 0.2**2
        assert ssd.iloc[0, 0] == pytest.approx(expected, abs=1e-12)

    def test_ranking_and_top_n(self):
        rng = np.random.default_rng(3)
        base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        prices = _panel(
            {
                "A": base,
                "CLOSE": base * (1 + rng.normal(0, 0.001, 300)),
                "FAR": base * (1 + rng.normal(0, 0.05, 300)),
            }
        )
        pairs = [("A", "CLOSE"), ("A", "FAR")]
        ranked = ssd_distances(prices, pairs)
        assert ranked.index[0] == ("A", "CLOSE")
        top = ssd_screen(prices, pairs, top_n=1)
        assert list(top.index) == [("A", "CLOSE")]

    def test_invalid_top_n_raises(self):
        prices = _panel({"A": np.linspace(100, 110, 50), "B": np.linspace(90, 80, 50)})
        with pytest.raises(ValueError, match="top_n"):
            ssd_screen(prices, [("A", "B")], top_n=0)

    def test_nonpositive_first_price_raises(self):
        prices = _panel({"A": np.array([0.0, 1.0, 2.0]), "B": np.array([1.0, 1.0, 1.0])})
        with pytest.raises(ValueError, match="positive"):
            ssd_distances(prices, [("A", "B")])
