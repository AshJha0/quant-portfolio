"""Bootstrap round trips (1e-10 contract, achieved at ~1e-15) and
basis-adjusted curve construction."""

import numpy as np
import pytest

from fx_rates.bootstrap import (
    basis_adjusted_curve,
    bootstrap_curve,
    curve_from_fx_forwards,
    deposit_rate_from_df,
    df_from_deposit,
    implied_basis_from_forwards,
    par_swap_rate,
    reprice_deposits,
    reprice_swaps,
)
from fx_rates.data import REGIMES, build_market_state, generate_market_quotes


class TestDeposits:
    def test_deposit_df_hand_check(self):
        # 4% simple for 0.5y: DF = 1/1.02
        assert df_from_deposit(0.04, 0.5) == pytest.approx(1 / 1.02, abs=1e-15)

    def test_deposit_rate_roundtrip(self):
        df = df_from_deposit(0.0335, 0.25)
        assert deposit_rate_from_df(df, 0.25) == pytest.approx(0.0335, abs=1e-15)

    def test_negative_deposit_rate_df_above_one(self):
        assert df_from_deposit(-0.005, 1.0) > 1.0

    def test_bad_tau_raises(self):
        with pytest.raises(ValueError):
            df_from_deposit(0.03, 0.0)


class TestBootstrapRoundTrip:
    @pytest.mark.parametrize("regime", REGIMES)
    @pytest.mark.parametrize("seed", [0, 42])
    def test_reprices_quotes_to_1e10(self, regime, seed):
        q = generate_market_quotes(regime, seed=seed)
        for deps, swps in [
            (q.domestic_deposits, q.domestic_swaps),
            (q.foreign_deposits, q.foreign_swaps),
        ]:
            curve = bootstrap_curve(deps, swps)
            assert reprice_deposits(curve, deps) < 1e-10
            assert reprice_swaps(curve, swps) < 1e-10

    @pytest.mark.parametrize("regime", REGIMES)
    def test_recovers_true_dfs_at_pillars(self, regime):
        q = generate_market_quotes(regime, seed=42)
        m = build_market_state(q)
        for boot, true in [
            (m.domestic_curve, q.true_domestic),
            (m.foreign_curve, q.true_foreign),
        ]:
            t = boot.times
            assert np.max(np.abs(boot.df(t) - true.df(t))) < 1e-12

    def test_par_swap_rate_definition(self, market):
        # fixed leg PV at par rate equals floating leg PV: 1 - DF(n)
        c = par_swap_rate(market.domestic_curve, 5)
        dfs = market.domestic_curve.df(np.arange(1.0, 6.0))
        assert c * dfs.sum() + dfs[-1] == pytest.approx(1.0, abs=1e-14)

    def test_no_deposits_raises(self):
        with pytest.raises(ValueError, match="deposit"):
            bootstrap_curve([], [(2.0, 0.03)])

    def test_non_integer_swap_maturity_raises(self):
        with pytest.raises(ValueError, match="integers"):
            bootstrap_curve([(1.0, 0.03)], [(2.5, 0.03)])

    def test_swaps_without_1y_deposit_raise(self):
        with pytest.raises(ValueError, match="1.0y"):
            bootstrap_curve([(0.25, 0.03)], [(2.0, 0.03)])


class TestBasisAdjustedCurve:
    def test_zero_basis_recovers_input_curve_exactly(self, market):
        adj = basis_adjusted_curve(
            market.foreign_curve, [(1.0, 0.0), (5.0, 0.0), (10.0, 0.0)]
        )
        t = np.linspace(0.05, 12.0, 200)
        assert np.max(np.abs(adj.df(t) - market.foreign_curve.df(t))) < 1e-14

    def test_empty_spreads_returns_same_curve(self, market):
        assert basis_adjusted_curve(market.foreign_curve, []) is market.foreign_curve

    def test_negative_basis_raises_foreign_dfs(self, market):
        adj = basis_adjusted_curve(market.foreign_curve, [(5.0, -0.0025)])
        # s < 0 => z_adj < z => DF_adj > DF
        assert adj.df(5.0) > market.foreign_curve.df(5.0)

    def test_spread_applied_exactly_at_pillar(self, market):
        adj = basis_adjusted_curve(market.foreign_curve, [(5.0, -0.0025)])
        assert adj.zero_rate(5.0) == pytest.approx(
            market.foreign_curve.zero_rate(5.0) - 0.0025, abs=1e-14
        )

    def test_implied_basis_matches_generator(self, quotes, market):
        implied = implied_basis_from_forwards(
            market.spot, market.domestic_curve, market.foreign_curve,
            quotes.fx_forward_quotes,
        )
        bt = np.array([t for t, _ in quotes.basis_spreads])
        bv = np.array([s for _, s in quotes.basis_spreads])
        for t, s in implied:
            assert s == pytest.approx(float(np.interp(t, bt, bv)), abs=1e-12)

    def test_curve_from_fx_forwards_matches_adjusted_curve(self, quotes, market):
        implied_curve = curve_from_fx_forwards(
            market.spot, market.domestic_curve, quotes.fx_forward_quotes
        )
        t = implied_curve.times
        assert np.max(
            np.abs(implied_curve.df(t) - market.foreign_curve_adjusted.df(t))
        ) < 1e-12

    def test_duplicate_spread_tenors_raise(self, market):
        with pytest.raises(ValueError, match="duplicate"):
            basis_adjusted_curve(market.foreign_curve, [(5.0, -0.001), (5.0, -0.002)])

    def test_bad_spot_raises(self, market):
        with pytest.raises(ValueError, match="spot"):
            curve_from_fx_forwards(-1.0, market.domestic_curve, [(1.0, 1.1)])
