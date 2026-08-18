"""Synthetic data tests: seeded reproducibility, planted alpha strength,
vol clustering, structural integrity."""

import numpy as np
import pandas as pd
import pytest

from eq_algo import (forward_returns, generate_daily_panel, ic_summary,
                     information_coefficient, momentum)


@pytest.fixture(scope="module")
def panel():
    return generate_daily_panel(n_stocks=80, n_days=700, seed=123)


def test_seeded_reproducibility():
    a = generate_daily_panel(n_stocks=10, n_days=300, seed=1)
    b = generate_daily_panel(n_stocks=10, n_days=300, seed=1)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    pd.testing.assert_frame_equal(a.volumes, b.volumes)
    c = generate_daily_panel(n_stocks=10, n_days=300, seed=2)
    assert not a.prices.equals(c.prices)


def test_planted_momentum_ic_in_target_band(panel):
    mom = momentum(panel.prices, 252, 21)
    fwd = forward_returns(panel.prices, 1)
    s = ic_summary(information_coefficient(mom, fwd))
    assert 0.02 < s["mean_ic"] < 0.07       # target IC ~ 0.03-0.05
    assert s["tstat_nw"] > 2.0


def test_planted_reversal_negative_relationship(panel):
    """Trailing 1-month return predicts *negatively* (so the reversal feature,
    which is minus that return, predicts positively)."""
    trailing = panel.prices / panel.prices.shift(21) - 1.0
    fwd = forward_returns(panel.prices, 1)
    s = ic_summary(information_coefficient(trailing, fwd))
    assert s["mean_ic"] < 0.0
    assert s["tstat_nw"] < -2.0


def test_vol_clustering_present(panel):
    """Lag-1 autocorrelation of squared returns is positive on average —
    the AR(1) log-vol process produces clustering."""
    r2 = (panel.returns.iloc[1:] ** 2)
    acs = [r2[c].autocorr(lag=1) for c in r2.columns]
    assert np.nanmean(acs) > 0.01
    assert np.mean([a > 0 for a in acs]) > 0.7


def test_structure_and_consistency(panel):
    assert panel.prices.shape == (700, 80)
    assert panel.prices.index.equals(panel.volumes.index)
    assert (panel.prices > 0).all().all()
    assert (panel.volumes > 0).all().all()
    pd.testing.assert_frame_equal(panel.returns, panel.prices.pct_change())
    assert (panel.adv_dollars > 0).all()


def test_generator_validation():
    with pytest.raises(ValueError):
        generate_daily_panel(n_stocks=1, n_days=100)
    with pytest.raises(ValueError):
        generate_daily_panel(n_stocks=10, n_days=1)
    with pytest.raises(ValueError, match="burn_in"):
        generate_daily_panel(n_stocks=10, n_days=100, burn_in=100)
