"""Backtest engine: no-lookahead proof, exact cost accounting, hand-checked P&L."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from eq_pairs.backtest import (
    COST_COLUMNS,
    ZERO_COSTS,
    CostModel,
    ExecutionConfig,
    align_pair,
    backtest_pair,
    backtest_portfolio,
)
from eq_pairs.data import business_index

# The hand-computed scenarios below were built for a next-close fill (f = 1),
# which is one valid point of the execution convention. Fill-fraction and
# close-auction behaviour is tested in TestExecutionConvention.
NEXT_CLOSE = ExecutionConfig(mode="next", fill_fraction=1.0)


def _bt(*args, **kwargs):
    kwargs.setdefault("execution", NEXT_CLOSE)
    return backtest_pair(*args, **kwargs)


def _series(vals) -> pd.Series:
    return pd.Series(np.asarray(vals, dtype=float), index=business_index(len(vals)))


def _target(vals, idx) -> pd.Series:
    return pd.Series(vals, index=idx, dtype=float)


class TestNoLookahead:
    """The lookahead detector: a spread engineered so same-day execution is
    profitable and correctly lagged execution loses. The engine MUST produce
    the losing number."""

    def _alternating_pair(self, n=41):
        s = np.where(np.arange(n) % 2 == 0, 3.0, -3.0)
        y = _series(100.0 + s)
        x = _series(np.full(n, 100.0))
        target = pd.Series(-np.sign(s), index=y.index)  # short rich, long cheap
        return y, x, target

    def test_engine_reports_the_lagged_losing_result(self):
        y, x, target = self._alternating_pair()
        lagged = _bt(y, x, target, beta=1.0, costs=ZERO_COSTS, gross=200.0)
        # cheat: shifting the signal forward lets the engine's t-1 lag see
        # the same-day value -> the profitable but impossible backtest
        cheat_target = target.shift(-1).fillna(0)
        cheat = _bt(y, x, cheat_target, beta=1.0, costs=ZERO_COSTS, gross=200.0)
        assert cheat.net_pnl > 0.0  # the trap is real: same-day info profits
        assert lagged.net_pnl < 0.0  # the engine refuses to take it

    def test_position_changes_one_bar_after_signal(self):
        n = 8
        y = _series(np.linspace(100, 107, n))
        x = _series(np.full(n, 100.0))
        target = _target([0, 0, 1, 1, 1, 0, 0, 0], y.index)
        res = _bt(y, x, target, beta=1.0, costs=ZERO_COSTS)
        pos = res.daily["position"]
        assert pos.iloc[2] == 0  # signal fires at t=2 ...
        assert pos.iloc[3] == 1  # ... position exists from t=3
        assert pos.iloc[5] == 1  # exit signal at t=5 ...
        assert pos.iloc[6] == 0  # ... flat from t=6

    def test_first_bar_never_trades(self):
        y = _series([100.0, 101.0, 102.0])
        x = _series([100.0, 100.0, 100.0])
        target = _target([1, 1, 1], y.index)
        res = _bt(y, x, target, beta=1.0, costs=ZERO_COSTS)
        assert res.daily["position"].iloc[0] == 0


class TestExecutionConvention:
    """Decide at close t, fill at p_t + f (p_{t+1} - p_t); close auction pays."""

    def test_fill_price_is_modelled_t_plus_1_price(self):
        # hand ledger: target goes long at t=1; fill on t=2 at
        # y: 101 + 0.5 (105 - 101) = 103, x: 50 + 0.5 (48 - 50) = 49
        y = _series([100.0, 101.0, 105.0, 107.0, 107.0])
        x = _series([50.0, 50.0, 48.0, 48.0, 48.0])
        tgt = _target([0, 1, 1, 1, 1], y.index)
        res = backtest_pair(y, x, tgt, beta=1.0, costs=ZERO_COSTS, gross=206.0,
                            execution=ExecutionConfig(fill_fraction=0.5))
        entry = res.ledger[res.ledger["date"] == y.index[2]].set_index("leg")
        assert entry.loc["y", "price"] == pytest.approx(103.0, abs=1e-12)
        assert entry.loc["x", "price"] == pytest.approx(49.0, abs=1e-12)
        q_y, q_x = 103.0 / 103.0, -103.0 / 49.0
        assert entry.loc["y", "shares"] == pytest.approx(q_y)
        assert entry.loc["x", "shares"] == pytest.approx(q_x)
        # day 2 P&L runs only from the fill to the close
        assert res.daily["gross_pnl"].iloc[2] == pytest.approx(q_y * 2.0 + q_x * -1.0)
        # day 4 forced exit at 107 + 0.5 (107 - 107)
        exit_ = res.ledger[res.ledger["date"] == y.index[4]].set_index("leg")
        assert exit_.loc["y", "price"] == pytest.approx(107.0)
        assert res.gross_pnl == pytest.approx(q_y * (107 - 103) + q_x * (48 - 49))

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_perfect_foresight_earns_nothing_at_next_close(self, seed):
        """s_t = sign(r_{t+1}) is perfect foresight of the next day. Filled at
        the close of t it would earn sum |r|; filled at the next close (f=1)
        it earns s_t r_{t+2}, mean zero. The test fails if the engine fills a
        decision at the close it was made on."""
        rng = np.random.default_rng(seed)
        n = 1500
        y = _series(1000.0 + np.cumsum(rng.normal(0.0, 1.0, n)))
        x = _series(np.full(n, 1000.0))
        tgt = _target(np.sign(np.append(np.diff(y.to_numpy()), 0.0)), y.index)
        res = backtest_pair(y, x, tgt, beta=1.0, costs=ZERO_COSTS, gross=2000.0,
                            execution=ExecutionConfig(fill_fraction=1.0))
        g = res.daily["gross_pnl"].to_numpy()[2:-1]
        se = g.std(ddof=1) / np.sqrt(g.size)
        assert abs(g.mean()) < 4.0 * se
        # the same decision filled at the close of t (close auction, no fee)
        # earns about E|dy| = 0.80 per day per unit: far outside 4 SE
        cheat = backtest_pair(y, x, tgt, beta=1.0, costs=ZERO_COSTS, gross=2000.0,
                              execution=ExecutionConfig(mode="close_auction",
                                                        auction_cost_bp=0.0))
        gc = cheat.daily["gross_pnl"].to_numpy()[2:-1]
        assert gc.mean() > 10.0 * se

    def test_fill_fraction_half_earns_half_of_the_foresight(self):
        rng = np.random.default_rng(9)
        n = 3000
        y = _series(1000.0 + np.cumsum(rng.normal(0.0, 1.0, n)))
        x = _series(np.full(n, 1000.0))
        tgt = _target(np.sign(np.append(np.diff(y.to_numpy()), 0.0)), y.index)
        half = backtest_pair(y, x, tgt, beta=1.0, costs=ZERO_COSTS, gross=2000.0,
                             execution=ExecutionConfig(fill_fraction=0.5))
        full = backtest_pair(y, x, tgt, beta=1.0, costs=ZERO_COSTS, gross=2000.0,
                             execution=ExecutionConfig(mode="close_auction",
                                                       auction_cost_bp=0.0))
        # a fill halfway along the next day's move captures half the foresight
        assert half.gross_pnl / full.gross_pnl == pytest.approx(0.5, abs=0.05)

    def test_close_auction_charges_exactly_the_auction_cost(self):
        y = _series([100.0, 101.0, 102.0, 101.0, 100.0])
        x = _series([50.0, 50.0, 50.0, 50.0, 50.0])
        tgt = _target([0, 1, 1, 0, 0], y.index)
        auc = dict(mode="close_auction")
        free = backtest_pair(y, x, tgt, beta=1.0, costs=ZERO_COSTS, gross=200.0,
                             execution=ExecutionConfig(**auc, auction_cost_bp=0.0))
        paid = backtest_pair(y, x, tgt, beta=1.0, costs=ZERO_COSTS, gross=200.0,
                             execution=ExecutionConfig(**auc, auction_cost_bp=25.0))
        # fills at the close: enter t=1 at 101/50 ($200 traded), exit t=3 at 101/50
        traded = 200.0 + (100.0 / 101.0 * 101.0 + 100.0)
        assert paid.daily["auction"].sum() == pytest.approx(25e-4 * traded, abs=1e-12)
        assert free.net_pnl - paid.net_pnl == pytest.approx(25e-4 * traded, abs=1e-12)
        assert paid.ledger.loc[0, "price"] == 101.0
        nxt = backtest_pair(y, x, tgt, beta=1.0, costs=ZERO_COSTS, gross=200.0)
        assert nxt.daily["auction"].sum() == 0.0

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
    def test_rejects_bad_fill_fraction(self, bad):
        with pytest.raises(ValueError):
            ExecutionConfig(fill_fraction=bad)

    def test_rejects_bad_mode_and_auction_cost(self):
        with pytest.raises(ValueError):
            ExecutionConfig(mode="same_close")
        with pytest.raises(ValueError):
            ExecutionConfig(auction_cost_bp=float("nan"))
        with pytest.raises(ValueError):
            ExecutionConfig(auction_cost_bp=-1.0)


class TestImpactAndFinancing:
    def test_sqrt_impact_per_leg_exact(self):
        y = _series([100.0, 100.0, 100.0, 100.0])
        x = _series([50.0, 50.0, 50.0, 50.0])
        tgt = _target([1, 1, 0, 0], y.index)
        costs = CostModel(0.0, 0.0, 0.0, impact_coef=0.7, adv_dollars=1e6)
        res = backtest_pair(y, x, tgt, beta=1.0, costs=costs, gross=200_000.0,
                            leg_vol=(0.02, 0.01))
        # each leg trades $100k = 10% of ADV: cost = N * 0.7 * sigma * sqrt(0.1)
        per_y = 1e5 * 0.7 * 0.02 * np.sqrt(0.1)
        per_x = 1e5 * 0.7 * 0.01 * np.sqrt(0.1)
        assert res.daily["impact"].sum() == pytest.approx(2 * (per_y + per_x), rel=1e-12)

    def test_impact_scales_as_size_to_the_three_halves(self):
        y = _series([100.0, 100.0, 100.0, 100.0])
        x = _series([50.0, 50.0, 50.0, 50.0])
        tgt = _target([1, 1, 0, 0], y.index)
        costs = CostModel(0.0, 0.0, 0.0, impact_coef=0.7, adv_dollars=1e6)
        small = backtest_pair(y, x, tgt, 1.0, costs=costs, gross=1e4, leg_vol=(0.02, 0.02))
        big = backtest_pair(y, x, tgt, 1.0, costs=costs, gross=4e4, leg_vol=(0.02, 0.02))
        assert big.daily["impact"].sum() == pytest.approx(8.0 * small.daily["impact"].sum())

    def test_financing_and_rebate_exact(self):
        y = _series([100.0, 100.0, 100.0, 100.0, 100.0])
        x = _series([50.0, 50.0, 50.0, 50.0, 50.0])
        tgt = _target([1, 1, 1, 0, 0], y.index)  # long y $100, short x $100
        costs = CostModel(0.0, 0.0, 0.0, financing_bps=252.0, short_rebate_bps=126.0)
        res = backtest_pair(y, x, tgt, 1.0, costs=costs, gross=200.0)
        # filled on t=1, exit filled on t=4: three nights at 100 * (1e-4 - 0.5e-4)
        assert res.daily["financing"].sum() == pytest.approx(3 * 100 * 0.5e-4, abs=1e-12)

    def test_net_identity_with_every_cost(self):
        rng = np.random.default_rng(12)
        n = 400
        y = _series(100 + np.cumsum(rng.normal(0, 1, n)))
        x = _series(80 + np.cumsum(rng.normal(0, 1, n)))
        tgt = _target(np.sign(np.sin(np.arange(n) / 7.0)), y.index)
        costs = CostModel(5.0, 2.0, 50.0, impact_coef=0.7, adv_dollars=2e6,
                          financing_bps=80.0, short_rebate_bps=20.0)
        for ex in (ExecutionConfig(), ExecutionConfig(mode="close_auction")):
            res = backtest_pair(y, x, tgt, 1.2, costs=costs, gross=1e5, execution=ex)
            d = res.daily
            np.testing.assert_allclose(
                d["net_pnl"], d["gross_pnl"] - d[COST_COLUMNS].sum(axis=1), atol=1e-9)
            assert res.trades["pnl"].sum() == pytest.approx(res.net_pnl, abs=1e-6)

    def test_cost_model_rejects_non_finite(self):
        with pytest.raises(ValueError):
            CostModel(impact_coef=float("nan"))
        with pytest.raises(ValueError):
            CostModel(adv_dollars=0.0)
        with pytest.raises(ValueError):
            CostModel(financing_bps=float("inf"))


class TestHandComputedScenario:
    """Three round trips with hand-computed P&L, matched exactly."""

    PY: ClassVar[list] = [100.0, 100.0, 102.0, 104.0, 100.0, 100.0, 98.0, 100.0, 100.0, 100.0, 101.0, 100.0]
    TGT: ClassVar[list] = [0, -1, -1, 0, 0, 1, 1, 0, 0, -1, -1, -1]

    def _run(self, costs):
        y = _series(self.PY)
        x = _series(np.full(len(self.PY), 100.0))
        return _bt(
            y, x, _target(self.TGT, y.index), beta=1.0, costs=costs,
            gross=200.0, sizing="dollar",
        )

    def test_gross_pnl_exact(self):
        res = self._run(ZERO_COSTS)
        # trade 1: short 100/102 sh at 102, exit 100 -> (100/102)*2
        # trade 2: long 100/98 sh at 98, exit 100 -> (100/98)*2
        # trade 3: short 100/101 sh at 101, exit 100 -> (100/101)*1
        expected = 200.0 / 102.0 + 200.0 / 98.0 + 100.0 / 101.0
        assert res.gross_pnl == pytest.approx(expected, abs=1e-9)
        assert res.net_pnl == pytest.approx(expected, abs=1e-9)

    def test_commissions_exact(self):
        costs = CostModel(cost_bps=10.0, slippage_bps=0.0, borrow_bps=0.0)
        res = self._run(costs)
        c = 10.0 / 1e4
        expected_comm = (
            c * 200.0  # t2 entry: two legs of $100 each
            + c * (100.0 / 102.0 * 100.0 + 100.0)  # t4 exit
            + c * 200.0  # t6 entry
            + c * (100.0 / 98.0 * 100.0 + 100.0)  # t8 exit
            + c * 200.0  # t10 entry
            + c * (100.0 / 101.0 * 100.0 + 100.0)  # t11 forced exit
        )
        assert res.daily["commission"].sum() == pytest.approx(expected_comm, abs=1e-9)
        gross = 200.0 / 102.0 + 200.0 / 98.0 + 100.0 / 101.0
        assert res.net_pnl == pytest.approx(gross - expected_comm, abs=1e-9)

    def test_trade_records(self):
        res = self._run(ZERO_COSTS)
        assert len(res.trades) == 3
        assert list(res.trades["direction"]) == [-1, 1, -1]
        assert list(res.trades["bars_held"]) == [2, 2, 1]
        assert list(res.trades["exit_reason"]) == ["signal", "signal", "end_of_sample"]
        np.testing.assert_allclose(
            res.trades["pnl"].to_numpy(),
            [200.0 / 102.0, 200.0 / 98.0, 100.0 / 101.0],
            atol=1e-9,
        )

    def test_trade_pnls_sum_to_net(self):
        costs = CostModel(cost_bps=7.0, slippage_bps=3.0, borrow_bps=100.0)
        res = self._run(costs)
        assert res.trades["pnl"].sum() == pytest.approx(res.net_pnl, abs=1e-9)


class TestCostAccounting:
    def _smoke(self, costs, seed=70):
        rng = np.random.default_rng(seed)
        n = 300
        s = np.sin(np.arange(n) / 5.0) * 4.0 + rng.normal(0, 0.3, n)
        y = _series(100.0 + s)
        x = _series(np.full(n, 100.0))
        target = pd.Series(
            np.select([s > 2.0, s < -2.0], [-1, 1], 0), index=y.index
        )
        return _bt(y, x, target, beta=1.0, costs=costs, gross=1000.0)

    def test_costs_reduce_pnl_by_exactly_ledger_sum(self):
        costs = CostModel(cost_bps=8.0, slippage_bps=4.0, borrow_bps=75.0)
        res = self._smoke(costs)
        assert len(res.ledger) > 4
        ledger_costs = float(res.ledger["commission"].sum() + res.ledger["slippage"].sum())
        borrow = float(res.daily["borrow"].sum())
        assert res.net_pnl == pytest.approx(res.gross_pnl - ledger_costs - borrow, abs=1e-9)
        # and the daily cost columns are the ledger, day by day
        assert res.daily["commission"].sum() == pytest.approx(
            res.ledger["commission"].sum(), abs=1e-9
        )
        assert res.daily["slippage"].sum() == pytest.approx(
            res.ledger["slippage"].sum(), abs=1e-9
        )

    def test_gross_pnl_invariant_to_costs(self):
        a = self._smoke(ZERO_COSTS)
        b = self._smoke(CostModel(cost_bps=20.0, slippage_bps=10.0, borrow_bps=200.0))
        assert a.gross_pnl == pytest.approx(b.gross_pnl, abs=1e-9)
        assert b.net_pnl < a.net_pnl

    def test_borrow_accrues_on_short_leg_exactly(self):
        py = [100.0, 102.0, 104.0, 100.0, 98.0, 100.0]
        y = _series(py)
        x = _series(np.full(6, 50.0))
        target = _target([-1, -1, -1, -1, 0, 0], y.index)
        costs = CostModel(cost_bps=0.0, slippage_bps=0.0, borrow_bps=252.0)  # 1e-4/day
        res = _bt(y, x, target, beta=1.0, costs=costs, gross=200.0)
        qy = 100.0 / 102.0  # |short shares| from t=1 entry at 102
        rate = 1e-4
        expected_by_day = [0.0, 0.0] + [rate * qy * p for p in py[2:]]
        np.testing.assert_allclose(res.daily["borrow"].to_numpy(), expected_by_day, atol=1e-12)
        assert res.net_pnl == pytest.approx(res.gross_pnl - sum(expected_by_day), abs=1e-9)

    def test_no_borrow_on_long_only_leg_days(self):
        y = _series([100.0, 98.0, 96.0, 100.0])
        x = _series(np.full(4, 100.0))
        target = _target([1, 1, 0, 0], y.index)  # long spread: long y, SHORT x
        costs = CostModel(cost_bps=0.0, slippage_bps=0.0, borrow_bps=252.0)
        res = _bt(y, x, target, beta=1.0, costs=costs, gross=200.0)
        # short x leg of $100 held into t2 -> borrow = 1e-4 * 100
        assert res.daily["borrow"].iloc[2] == pytest.approx(1e-4 * 100.0, abs=1e-12)


class TestNeutralityAndSizing:
    def test_dollar_neutral_at_entry(self):
        rng = np.random.default_rng(71)
        n = 100
        y = _series(100 + np.cumsum(rng.normal(0, 1, n)))
        x = _series(80 + np.cumsum(rng.normal(0, 1, n)))
        tgt = np.zeros(n)
        tgt[10:20] = 1
        tgt[40:55] = -1
        res = _bt(y, x, _target(tgt, y.index), beta=1.3, costs=ZERO_COSTS,
                            gross=1000.0, sizing="dollar")
        for entry_date in res.trades["entry_date"]:
            net = res.daily.loc[entry_date, "net_exposure"]
            gross = res.daily.loc[entry_date, "gross_exposure"]
            assert net == pytest.approx(0.0, abs=1e-9)
            assert gross == pytest.approx(1000.0, abs=1e-9)

    def test_beta_neutral_share_ratio(self):
        y = _series([100.0, 100.0, 102.0, 100.0])
        x = _series([50.0, 50.0, 50.0, 50.0])
        res = _bt(y, x, _target([-1, -1, -1, -1], y.index), beta=1.5,
                            costs=ZERO_COSTS, gross=1000.0, sizing="beta")
        row = res.daily.iloc[1]
        assert row["q_x"] == pytest.approx(-1.5 * row["q_y"], abs=1e-12)


class TestEngineMechanics:
    def test_zero_trades(self):
        y = _series(np.linspace(100, 110, 50))
        x = _series(np.linspace(100, 90, 50))
        res = _bt(y, x, _target(np.zeros(50), y.index), beta=1.0)
        assert len(res.ledger) == 0
        assert len(res.trades) == 0
        assert (res.daily["net_pnl"] == 0.0).all()
        assert (res.daily["position"] == 0).all()

    def test_flip_creates_two_round_trips(self):
        y = _series([100.0, 103.0, 97.0, 96.0, 100.0, 100.0])
        x = _series(np.full(6, 100.0))
        res = _bt(y, x, _target([0, -1, 1, 1, 0, 0], y.index), beta=1.0,
                            costs=ZERO_COSTS, gross=200.0)
        assert len(res.trades) == 2
        assert list(res.trades["direction"]) == [-1, 1]
        assert len(res.ledger) == 8  # 2 entry + (2 close + 2 open) + 2 close

    def test_close_at_end_forces_flat(self):
        y = _series([100.0, 103.0, 104.0, 105.0])
        x = _series(np.full(4, 100.0))
        res = _bt(y, x, _target([-1, -1, -1, -1], y.index), beta=1.0,
                            costs=ZERO_COSTS)
        assert res.daily["position"].iloc[-1] == 0
        assert res.trades["exit_reason"].iloc[0] == "end_of_sample"

    def test_nan_target_treated_as_flat(self):
        y = _series([100.0, 101.0, 102.0, 103.0])
        x = _series(np.full(4, 100.0))
        tgt = pd.Series([np.nan, 1.0, np.nan, 0.0], index=y.index)
        res = _bt(y, x, tgt, beta=1.0, costs=ZERO_COSTS)
        assert list(res.daily["position"]) == [0, 0, 1, 0]

    def test_invalid_target_raises(self):
        y = _series([100.0, 101.0, 102.0])
        x = _series(np.full(3, 100.0))
        with pytest.raises(ValueError, match="invalid values"):
            _bt(y, x, _target([0, 2, 0], y.index), beta=1.0)

    def test_index_mismatch_raises(self):
        y = _series([100.0, 101.0, 102.0])
        x = pd.Series([100.0, 100.0, 100.0], index=business_index(3, "2020-01-01"))
        with pytest.raises(ValueError, match="indices differ"):
            _bt(y, x, _target([0, 0, 0], y.index), beta=1.0)

    def test_nan_price_raises(self):
        y = _series([100.0, np.nan, 102.0])
        x = _series(np.full(3, 100.0))
        with pytest.raises(ValueError, match="NaN"):
            _bt(y, x, _target([0, 0, 0], y.index), beta=1.0)


class TestAlignPair:
    def test_short_gap_ffilled(self):
        idx = business_index(10)
        y = pd.Series(np.arange(100.0, 110.0), index=idx)
        y.iloc[4:6] = np.nan
        x = pd.Series(np.full(10, 50.0), index=idx)
        ya, _xa = align_pair(y, x, policy="ffill", limit=5)
        assert ya.notna().all()
        assert ya.iloc[4] == ya.iloc[3]  # stale-filled

    def test_long_gap_raises(self):
        idx = business_index(20)
        y = pd.Series(np.arange(100.0, 120.0), index=idx)
        y.iloc[5:14] = np.nan
        x = pd.Series(np.full(20, 50.0), index=idx)
        with pytest.raises(ValueError, match="gap too long"):
            align_pair(y, x, policy="ffill", limit=5)

    def test_drop_policy(self):
        idx = business_index(10)
        y = pd.Series(np.arange(100.0, 110.0), index=idx)
        y.iloc[3] = np.nan
        x = pd.Series(np.full(10, 50.0), index=idx)
        ya, xa = align_pair(y, x, policy="drop")
        assert len(ya) == 9 and ya.index.equals(xa.index)

    def test_invalid_policy_raises(self):
        y = _series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="policy"):
            align_pair(y, y, policy="interpolate")


class TestPortfolio:
    def _two_pair_setup(self):
        rng = np.random.default_rng(72)
        n = 200
        idx = business_index(n)
        panel = pd.DataFrame(
            {
                "A": 100 + np.sin(np.arange(n) / 4.0) * 3 + rng.normal(0, 0.2, n),
                "B": np.full(n, 100.0),
                "C": 50 + np.cos(np.arange(n) / 5.0) * 2 + rng.normal(0, 0.2, n),
                "D": np.full(n, 50.0),
            },
            index=idx,
        )
        sa = panel["A"] - panel["B"]
        sc = panel["C"] - panel["D"]
        targets = {
            ("A", "B"): pd.Series(np.select([sa > 2, sa < -2], [-1, 1], 0), index=idx),
            ("C", "D"): pd.Series(np.select([sc > 1, sc < -1], [-1, 1], 0), index=idx),
        }
        betas = {("A", "B"): 1.0, ("C", "D"): 1.0}
        return panel, targets, betas

    def test_daily_aggregation_is_sum_of_pairs(self):
        panel, targets, betas = self._two_pair_setup()
        port = backtest_portfolio(panel, targets, betas, costs=ZERO_COSTS)
        assert len(port.pairs) == 2
        manual = sum(p.daily["net_pnl"] for p in port.pairs)
        np.testing.assert_allclose(
            port.daily["net_pnl"].to_numpy(), manual.to_numpy(), atol=1e-9
        )

    def test_attribution_rows_and_totals(self):
        panel, targets, betas = self._two_pair_setup()
        port = backtest_portfolio(panel, targets, betas)
        att = port.attribution()
        assert set(att.index) == {"A/B", "C/D"}
        assert att["net_pnl"].sum() == pytest.approx(port.net_pnl, abs=1e-9)

    def test_single_pair_portfolio(self):
        panel, targets, betas = self._two_pair_setup()
        one = {("A", "B"): targets[("A", "B")]}
        port = backtest_portfolio(panel, one, betas)
        assert len(port.pairs) == 1
        assert port.net_pnl == pytest.approx(port.pairs[0].net_pnl, abs=1e-12)

    def test_empty_portfolio_raises(self):
        panel, _, betas = self._two_pair_setup()
        with pytest.raises(ValueError, match="empty"):
            backtest_portfolio(panel, {}, betas)
