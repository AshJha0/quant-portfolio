"""Synthetic generators: seeding, ground-truth recovery, correlation structure."""

import numpy as np
import pandas as pd
import pytest

import fx_pairs as fp
from fx_pairs.data import synthetic as syn


class TestSeeding:
    def test_same_seed_same_data(self):
        a1, b1, _ = syn.make_cointegrated_pair(n=300, seed=42)
        a2, b2, _ = syn.make_cointegrated_pair(n=300, seed=42)
        assert np.array_equal(a1.values, a2.values)
        assert np.array_equal(b1.values, b2.values)

    def test_different_seed_different_data(self):
        a1, _, _ = syn.make_cointegrated_pair(n=300, seed=42)
        a2, _, _ = syn.make_cointegrated_pair(n=300, seed=43)
        assert not np.array_equal(a1.values, a2.values)

    def test_business_day_index(self):
        p1, _, _ = syn.make_cointegrated_pair(n=300, seed=0)
        assert isinstance(p1.index, pd.DatetimeIndex)
        assert (p1.index.dayofweek < 5).all()
        assert len(p1) == 300


class TestCointegratedPair:
    def test_ground_truth_beta_recovered(self):
        p1, p2, truth = syn.make_cointegrated_pair(n=2500, beta=1.3, alpha=0.1,
                                                   seed=5)
        eg = fp.engle_granger(np.log(p1.values), np.log(p2.values))
        assert eg.cointegrated
        assert eg.beta == pytest.approx(truth["beta"], abs=0.05)

    def test_spread_is_stationary_ou(self):
        p1, p2, truth = syn.make_cointegrated_pair(n=2500, beta=1.0, kappa=20.0,
                                                   sigma_ou=0.05, seed=6)
        s = fp.log_spread(p1, p2, truth["beta"], truth["alpha"])
        fit = fp.fit_ou_ols(s.values)
        assert fit.kappa == pytest.approx(truth["kappa"], rel=0.4)
        assert fp.adf_test(s.values).reject("1%")


class TestOUSimulator:
    def test_stationary_moments(self):
        kappa, sigma = 30.0, 0.06
        s = syn.simulate_ou(20000, kappa, 0.02, sigma, seed=8)
        stat_sd = sigma / np.sqrt(2 * kappa)
        assert np.mean(s) == pytest.approx(0.02, abs=4 * stat_sd / np.sqrt(200))
        assert np.std(s) == pytest.approx(stat_sd, rel=0.1)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            syn.simulate_ou(100, -1.0, 0.0, 0.05, seed=0)
        with pytest.raises(ValueError):
            syn.simulate_ou(100, 1.0, 0.0, 0.0, seed=0)


@pytest.fixture(scope="module")
def panel():
    return syn.make_two_block_panel(n=1500, n_flips=3, flip_len=60, seed=2)


class TestTwoBlockPanel:
    def _mean_corr(self, rets, a, b):
        vals = [rets[x].corr(rets[y]) for x in a for y in b if x != y]
        return float(np.mean(vals))

    def test_intra_block_correlation_high(self, panel):
        legs, regime = panel
        rets = np.log(legs).diff().dropna()
        calm = rets[regime.iloc[1:].values == 0]
        assert self._mean_corr(calm, ("AUD", "NZD", "CAD"), ("AUD", "NZD", "CAD")) > 0.7
        assert self._mean_corr(calm, ("JPY", "CHF"), ("JPY", "CHF")) > 0.7

    def test_cross_block_low_in_calm(self, panel):
        legs, regime = panel
        rets = np.log(legs).diff().dropna()
        calm = rets[regime.iloc[1:].values == 0]
        cross = self._mean_corr(calm, ("AUD", "NZD", "CAD"), ("JPY", "CHF"))
        assert abs(cross) < 0.35

    def test_cross_block_flips_negative_in_riskoff(self, panel):
        legs, regime = panel
        rets = np.log(legs).diff().dropna()
        off = rets[regime.iloc[1:].values == 1]
        cross = self._mean_corr(off, ("AUD", "NZD", "CAD"), ("JPY", "CHF"))
        assert cross < -0.3

    def test_flips_detectable_by_rolling_correlation(self, panel):
        """A 40d rolling AUD-JPY leg correlation must visibly drop inside the
        risk-off windows: the flip is detectable, not just definitional."""
        legs, regime = panel
        rets = np.log(legs).diff()
        roll = rets["AUD"].rolling(40).corr(rets["JPY"])
        in_off = roll[regime.shift(-0).values == 1].dropna()
        in_calm = roll[regime.values == 0].dropna()
        assert in_off.mean() < in_calm.mean() - 0.3

    def test_regime_series_matches_spec(self, panel):
        _, regime = panel
        # 3 flips of 60 days each
        assert int(regime.sum()) == 3 * 60
        changes = np.abs(np.diff(regime.values)).sum()
        assert changes == 6  # 3 on + 3 off


class TestFloorBreak:
    def test_prebreak_spread_tiny_vol_then_jump(self):
        p1, p2, meta = syn.make_floor_then_break(seed=3)
        s = fp.log_spread(p1, p2, meta["beta"]).values
        bi = meta["break_idx"]
        assert np.std(s[:bi]) < 0.01
        jump_realised = s[bi] - s[bi - 1]
        assert jump_realised == pytest.approx(meta["jump"], abs=1e-12)
        assert np.std(np.diff(s[bi:])) > 3 * np.std(np.diff(s[:bi]))

    def test_formation_scan_scores_peg_as_excellent(self):
        """The trap: pre-break data looks like textbook cointegration."""
        p1, p2, _ = syn.make_floor_then_break(seed=3)
        eg = fp.engle_granger(np.log(p1.values[:500]), np.log(p2.values[:500]))
        assert eg.cointegrated
        assert not eg.degenerate


class TestDepositRates:
    def test_levels_and_persistence(self):
        idx = syn.business_days(1000)
        rates = syn.make_deposit_rate_panel(idx, {"HY": 0.08, "USD": 0.01},
                                            vol=0.002, seed=5)
        assert rates["HY"].mean() == pytest.approx(0.08, abs=0.01)
        assert rates["USD"].mean() == pytest.approx(0.01, abs=0.01)
        # persistent differential: never inverts
        assert (rates["HY"] - rates["USD"]).min() > 0.0

    def test_floor_at_minus_one_percent(self):
        idx = syn.business_days(500)
        rates = syn.make_deposit_rate_panel(idx, {"CHF": -0.0075}, vol=0.004,
                                            seed=6)
        assert rates["CHF"].min() >= -0.01


class TestPeggedPair:
    def test_zero_volatility(self):
        peg = syn.make_pegged_pair(n=300)
        assert peg.std() == 0.0
        assert len(peg) == 300
