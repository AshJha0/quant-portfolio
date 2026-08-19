"""parameter_grid: pivoted shape, excluded fast>=slow combinations."""

import numpy as np
import pandas as pd

from eq_signal_backtest.sensitivity import parameter_grid


def _synthetic_prices(n=120, seed=3):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    prices = 100 * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=pd.bdate_range("2022-01-01", periods=n))


def test_grid_shape_and_orientation():
    prices = _synthetic_prices()
    fast_range = range(2, 6)  # 2,3,4,5
    slow_range = range(10, 22, 4)  # 10,14,18
    grid = parameter_grid(prices, fast_range, slow_range, cost_bps=0.0)
    assert isinstance(grid, pd.DataFrame)
    assert set(grid.index) <= set(fast_range)
    assert set(grid.columns) <= set(slow_range)
    assert grid.index.name == "fast"
    assert grid.columns.name == "slow"


def test_grid_excludes_fast_ge_slow_combinations():
    prices = _synthetic_prices()
    fast_range = [3, 5, 10]
    slow_range = [5, 10]
    grid = parameter_grid(prices, fast_range, slow_range, cost_bps=0.0)
    # fast=5,slow=5 was skipped (equal) but fast=5 has a valid slow=10 cell,
    # so the row exists with an explicit NaN at the invalid cell.
    assert pd.isna(grid.loc[5, 5])
    # fast=10 has NO valid slow in slow_range (10 >= both 5 and 10), so the
    # whole row is absent from the pivoted grid rather than all-NaN.
    assert 10 not in grid.index
    # a genuinely valid combination must be present and numeric
    assert pd.notna(grid.loc[3, 5])
    assert pd.notna(grid.loc[5, 10])


def test_grid_all_invalid_combinations_returns_empty():
    prices = _synthetic_prices()
    grid = parameter_grid(prices, fast_range=[10, 20], slow_range=[5, 8], cost_bps=0.0)
    assert grid.empty


def test_grid_values_are_sharpe_ratios_matching_direct_backtest():
    from eq_signal_backtest.engine import run_backtest
    from eq_signal_backtest.signals import ma_crossover_signal

    prices = _synthetic_prices()
    grid = parameter_grid(prices, fast_range=[4], slow_range=[12], cost_bps=3.0)
    sig = ma_crossover_signal(prices, 4, 12)
    res = run_backtest(prices, sig, cost_bps=3.0)
    assert grid.loc[4, 12] == res.stats["sharpe"] or (
        pd.isna(grid.loc[4, 12]) and pd.isna(res.stats["sharpe"])
    )
