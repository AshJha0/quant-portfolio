"""Risk-report tests: exact partitions and P&L attribution identities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eq_regime.risk import (
    flip_aftermath,
    per_regime_stats,
    regime_runs,
    transition_attribution,
)


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    net = pd.Series(rng.standard_normal(300) * 0.01, index=idx)
    regimes = pd.Series(
        ["bull"] * 120 + ["bear"] * 60 + ["transition"] * 40 + ["bull"] * 80, index=idx
    )
    return net, regimes


def test_per_regime_counts_partition_sample(sample):
    net, regimes = sample
    table = per_regime_stats(net, regimes)
    regime_rows = table.drop(index="TOTAL")
    assert regime_rows["days"].sum() == table.loc["TOTAL", "days"] == len(net)


def test_per_regime_pnl_sums_to_total(sample):
    net, regimes = sample
    table = per_regime_stats(net, regimes)
    assert table.drop(index="TOTAL")["total_pnl"].sum() == pytest.approx(
        table.loc["TOTAL", "total_pnl"], abs=1e-12
    )
    assert table.loc["TOTAL", "total_pnl"] == pytest.approx(net.sum(), abs=1e-12)


def test_per_regime_stats_values(sample):
    net, regimes = sample
    table = per_regime_stats(net, regimes)
    bear = net[regimes == "bear"]
    assert table.loc["bear", "ann_return"] == pytest.approx(bear.mean() * 252)
    assert table.loc["bear", "ann_vol"] == pytest.approx(bear.std(ddof=1) * np.sqrt(252))
    assert table.loc["bear", "days"] == 60


def test_attribution_identity(sample):
    net, regimes = sample
    att = transition_attribution(net, regimes)
    regime_rows = att.drop(index="TOTAL")
    assert regime_rows["pnl"].sum() == pytest.approx(att.loc["TOTAL", "pnl"], abs=1e-12)
    assert att.loc["TOTAL", "pnl"] == pytest.approx(net.sum(), abs=1e-12)
    # 3 flips in the constructed path
    assert att.loc["TOTAL", "flip_days"] == 3
    assert regime_rows["flip_days"].sum() == 3


def test_flip_aftermath_constructed():
    idx = pd.bdate_range("2020-01-01", periods=10)
    net = pd.Series([0.01] * 5 + [-0.02] * 5, index=idx)
    regimes = pd.Series(["bull"] * 5 + ["bear"] * 5, index=idx)
    out = flip_aftermath(net, regimes, k=3)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["from_regime"] == "bull" and row["to_regime"] == "bear"
    assert row["pnl_next_3d"] == pytest.approx(0.98**3 - 1)
    assert row["date"] == idx[5]


def test_flip_aftermath_window_truncated_at_end():
    idx = pd.bdate_range("2020-01-01", periods=6)
    net = pd.Series([0.0, 0.0, 0.0, 0.0, 0.01, 0.01], index=idx)
    regimes = pd.Series(["a", "a", "a", "a", "b", "b"], index=idx)
    out = flip_aftermath(net, regimes, k=10)
    assert out.iloc[0]["pnl_next_10d"] == pytest.approx(1.01**2 - 1)


def test_regime_runs(sample):
    _, regimes = sample
    runs = regime_runs(regimes)
    assert list(runs["regime"]) == ["bull", "bear", "transition", "bull"]
    assert list(runs["days"]) == [120, 60, 40, 80]
    assert runs["days"].sum() == len(regimes)


def test_validation():
    idx = pd.bdate_range("2020-01-01", periods=5)
    net = pd.Series(0.0, index=idx)
    bad = pd.Series("a", index=pd.bdate_range("2021-01-01", periods=5))
    with pytest.raises(ValueError, match="same index"):
        per_regime_stats(net, bad)
    with pytest.raises(ValueError, match="same index"):
        transition_attribution(net, bad)
    good = pd.Series("a", index=idx)
    with pytest.raises(ValueError, match="k must be"):
        flip_aftermath(net, good, k=0)
