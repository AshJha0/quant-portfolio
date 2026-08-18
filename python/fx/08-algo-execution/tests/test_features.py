"""Bar construction and feature causality (point-in-time discipline)."""

import numpy as np
import pandas as pd
import pytest

from fx_algo import (
    build_bars,
    carry_feature,
    feature_matrix,
    generate_daily_panel,
    generate_ticks,
    london_open_breakout,
    momentum,
    reversion_to_session_mean,
)


def toy_ticks() -> pd.DataFrame:
    # two 1h bars: [0,1) has mids 1.10, 1.12, 1.09; [1,2) has 1.11, 1.15
    return pd.DataFrame(
        {
            "time_hours": [0.0, 0.4, 0.9, 1.0, 1.5],
            "mid": [1.10, 1.12, 1.09, 1.11, 1.15],
        }
    )


def test_bar_construction_exact_on_toy_ticks():
    bars = build_bars(toy_ticks(), 1.0)
    assert len(bars) == 2
    assert list(bars.index) == [1.0, 2.0]
    b0, b1 = bars.iloc[0], bars.iloc[1]
    assert (b0["open"], b0["high"], b0["low"], b0["close"]) == (1.10, 1.12, 1.09, 1.09)
    assert b0["twap_mid"] == pytest.approx((1.10 + 1.12 + 1.09) / 3)
    assert b0["n_ticks"] == 3
    assert (b1["open"], b1["high"], b1["low"], b1["close"]) == (1.11, 1.15, 1.11, 1.15)


def test_boundary_tick_opens_next_bar():
    bars = build_bars(toy_ticks(), 1.0)
    # the tick at exactly t=1.0 belongs to the second bar
    assert bars.iloc[1]["open"] == 1.11


def test_bar_hour_and_day_columns():
    ticks = generate_ticks(n_days=2, seed=0)
    bars = build_bars(ticks, 1.0)
    assert bars["hour"].iloc[0] == 1.0
    assert bars["hour"].iloc[23] == 0.0  # bar ending at 24:00 -> hour 0
    assert bars["day"].iloc[0] == 0
    assert bars["day"].iloc[23] == 0  # bar ending exactly at 24.0 is day 0
    assert bars["day"].iloc[24] == 1


def test_build_bars_invalid():
    with pytest.raises(ValueError):
        build_bars(toy_ticks(), 0.0)
    with pytest.raises(ValueError):
        build_bars(toy_ticks().iloc[:0], 1.0)


def test_momentum_exact_values():
    ticks = toy_ticks()
    bars = build_bars(ticks, 1.0)
    m = momentum(bars, 1)
    assert np.isnan(m.iloc[0])
    assert m.iloc[1] == pytest.approx(1.15 / 1.09 - 1.0)
    with pytest.raises(ValueError):
        momentum(bars, 0)


def test_reversion_exact_values():
    bars = build_bars(toy_ticks(), 1.0)
    rev = reversion_to_session_mean(bars)
    tw0 = (1.10 + 1.12 + 1.09) / 3
    tw1 = (1.11 + 1.15) / 2
    assert rev.iloc[0] == pytest.approx((tw0 - 1.09) / 1.09)
    assert rev.iloc[1] == pytest.approx(((tw0 + tw1) / 2 - 1.15) / 1.15)


def test_breakout_logic_on_constructed_day():
    # Asia bars (hours 1..7) with range [1.09, 1.11]; bar ending hour 8
    # closes above -> +1; bar ending hour 9 closes below -> -1.
    times, mids = [], []
    for h in range(7):  # ticks in [0,7): range 1.09..1.11
        times += [h + 0.1, h + 0.5]
        mids += [1.09, 1.11]
    times += [7.5, 8.5]
    mids += [1.13, 1.05]
    bars = build_bars(pd.DataFrame({"time_hours": times, "mid": mids}), 1.0)
    bo = london_open_breakout(bars)
    assert bo.loc[8.0] == 1.0
    assert bo.loc[9.0] == -1.0
    assert (bo.loc[bars["hour"] <= 7.0] == 0.0).all()


def test_carry_feature_broadcast():
    ticks = generate_ticks(n_days=3, seed=0)
    bars = build_bars(ticks, 1.0)
    panel = generate_daily_panel(3, seed=0)
    c = carry_feature(bars, panel)
    for d in range(3):
        vals = c[bars["day"] == d].unique()
        assert len(vals) == 1
        assert vals[0] == pytest.approx(panel["carry"].loc[d])


def test_feature_matrix_columns():
    ticks = generate_ticks(n_days=3, seed=0)
    bars = build_bars(ticks, 1.0)
    fm = feature_matrix(bars, generate_daily_panel(3, seed=0))
    assert list(fm.columns) == ["mom_1", "mom_4", "reversion", "breakout", "carry"]
    fm2 = feature_matrix(bars)  # no panel -> zero carry
    assert (fm2["carry"] == 0.0).all()


def test_pit_mutation_leaves_past_features_unchanged():
    """The point-in-time discipline test: mutating future ticks must not
    change any feature value at or before the cutoff."""
    ticks = generate_ticks(n_days=6, seed=11)
    panel = generate_daily_panel(6, seed=11)
    cutoff = 3 * 24.0  # end of day 2

    bars = build_bars(ticks, 1.0)
    fm = feature_matrix(bars, panel)

    mutated = ticks.copy()
    future = mutated["time_hours"] >= cutoff
    assert future.any()
    rng = np.random.default_rng(0)
    mutated.loc[future, "mid"] += 0.05 * rng.standard_normal(int(future.sum()))
    fm_mut = feature_matrix(build_bars(mutated, 1.0), panel)

    past = fm.index <= cutoff
    pd.testing.assert_frame_equal(fm.loc[past], fm_mut.loc[past])
    # sanity: the future did change
    assert not fm.loc[~past].equals(fm_mut.loc[~past])


def test_pit_mutation_bar_level():
    ticks = generate_ticks(n_days=4, seed=5)
    bars = build_bars(ticks, 1.0)
    mutated = ticks.copy()
    mutated.loc[mutated["time_hours"] >= 48.0, "mid"] = 9.99
    bars_mut = build_bars(mutated, 1.0)
    past = bars.index <= 48.0
    pd.testing.assert_frame_equal(bars.loc[past], bars_mut.loc[past])
