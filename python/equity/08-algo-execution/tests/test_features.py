"""Feature tests: point-in-time discipline (mutation tests), hand-computed
values on tiny series, cross-sectional utilities, winsorisation bounds."""

import numpy as np
import pandas as pd
import pytest

from eq_algo import (cs_rank, cs_zscore, generate_daily_panel, ma_crossover,
                     momentum, realized_vol, rsi, short_term_reversal,
                     turnover_zscore, winsorize)


@pytest.fixture(scope="module")
def panel():
    return generate_daily_panel(n_stocks=20, n_days=320, seed=11)


CUT = 300  # mutation cutoff (row position)

FEATURE_FUNCS = {
    "momentum": lambda p, v: momentum(p, 252, 21),
    "reversal": lambda p, v: short_term_reversal(p, 21),
    "realized_vol": lambda p, v: realized_vol(p, 63),
    "ma_crossover": lambda p, v: ma_crossover(p, 20, 100),
    "rsi": lambda p, v: rsi(p, 14),
    "turnover_zscore": lambda p, v: turnover_zscore(v, 63),
}


@pytest.mark.parametrize("name", list(FEATURE_FUNCS))
def test_point_in_time_mutation(panel, name):
    """Gold-standard leakage test: mutating data strictly after t leaves the
    feature at all dates <= t bit-identical."""
    func = FEATURE_FUNCS[name]
    base = func(panel.prices, panel.volumes)
    prices2 = panel.prices.copy()
    volumes2 = panel.volumes.copy()
    rng = np.random.default_rng(0)
    prices2.iloc[CUT + 1:] = prices2.iloc[CUT + 1:].to_numpy() * \
        rng.uniform(0.2, 5.0, prices2.iloc[CUT + 1:].shape)
    volumes2.iloc[CUT + 1:] = volumes2.iloc[CUT + 1:].to_numpy() * \
        rng.uniform(0.2, 5.0, volumes2.iloc[CUT + 1:].shape)
    mutated = func(prices2, volumes2)
    pd.testing.assert_frame_equal(base.iloc[:CUT + 1], mutated.iloc[:CUT + 1])


def test_momentum_hand_computed():
    p = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]})
    m = momentum(p, lookback=3, skip=1)
    # t=4: P_{t-1}/P_{t-3} - 1 = 4/2 - 1 = 1
    assert m["A"].iloc[4] == pytest.approx(1.0, abs=1e-12)
    assert m["A"].iloc[3] == pytest.approx(3.0 / 1.0 - 1.0, abs=1e-12)
    assert np.isnan(m["A"].iloc[2])


def test_reversal_hand_computed():
    p = pd.DataFrame({"A": [1.0, 2.0, 4.0]})
    r = short_term_reversal(p, lookback=2)
    assert r["A"].iloc[2] == pytest.approx(-3.0, abs=1e-12)
    assert np.isnan(r["A"].iloc[1])


def test_realized_vol_hand_computed():
    p = pd.DataFrame({"A": [1.0, np.exp(0.01), np.exp(0.03)]})
    v = realized_vol(p, window=2, annualize=True)
    expected = np.std([0.01, 0.02], ddof=1) * np.sqrt(252.0)
    assert v["A"].iloc[2] == pytest.approx(expected, rel=1e-12)


def test_ma_crossover_hand_computed():
    p = pd.DataFrame({"A": [2.0, 4.0, 6.0]})
    m = ma_crossover(p, fast=1, slow=2)
    # t=2: 6 / ((4+6)/2) - 1 = 0.2
    assert m["A"].iloc[2] == pytest.approx(0.2, abs=1e-12)


def test_rsi_hand_computed():
    p = pd.DataFrame({"A": [1.0, 2.0, 3.0, 2.0]})
    r = rsi(p, window=2)
    # t=2: gains (1,1) avg 1, losses 0 -> 100
    assert r["A"].iloc[2] == pytest.approx(100.0, abs=1e-12)
    # t=3: gains (1,0) avg .5, losses (0,1) avg .5 -> 50
    assert r["A"].iloc[3] == pytest.approx(50.0, abs=1e-12)


def test_rsi_flat_prices_neutral():
    p = pd.DataFrame({"A": [5.0] * 10})
    r = rsi(p, window=3)
    assert (r["A"].iloc[3:] == 50.0).all()
    assert np.isnan(r["A"].iloc[2])  # first diff is NaN -> warm-up


def test_cs_rank_hand_computed():
    df = pd.DataFrame([[3.0, 1.0, 2.0]], columns=list("ABC"))
    out = cs_rank(df)
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [1.0, 1 / 3, 2 / 3])


def test_cs_zscore_hand_computed():
    df = pd.DataFrame([[1.0, 2.0, 3.0]], columns=list("ABC"))
    out = cs_zscore(df)
    e = 1.0 / np.sqrt(2.0 / 3.0)
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [-e, 0.0, e], atol=1e-12)


def test_cs_zscore_degenerate_row_is_nan():
    df = pd.DataFrame([[2.0, 2.0, 2.0]], columns=list("ABC"))
    assert cs_zscore(df).iloc[0].isna().all()


def test_winsorize_bounds():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.standard_normal((50, 200)))
    w = winsorize(df, 0.05, 0.95)
    lo = df.quantile(0.05, axis=1)
    hi = df.quantile(0.95, axis=1)
    assert (w.max(axis=1) <= hi + 1e-12).all()
    assert (w.min(axis=1) >= lo - 1e-12).all()
    # interior values untouched
    inside = df.gt(lo, axis=0) & df.lt(hi, axis=0)
    pd.testing.assert_frame_equal(w[inside], df[inside])


def test_winsorize_preserves_nan():
    df = pd.DataFrame([[1.0, np.nan, 2.0, 3.0, 10.0]])
    w = winsorize(df, 0.0, 0.75)
    assert np.isnan(w.iloc[0, 1])


def test_turnover_zscore_hand_computed():
    v = pd.DataFrame({"A": [1.0, np.e, np.e**2]})
    z = turnover_zscore(v, window=2)
    # log volumes 0,1,2; t=1: (1-0.5)/std([0,1],ddof=1) = 0.5/0.7071...
    assert z["A"].iloc[1] == pytest.approx(0.5 / np.std([0, 1], ddof=1), rel=1e-12)
    assert np.isnan(z["A"].iloc[0])


@pytest.mark.parametrize("bad_call", [
    lambda p: momentum(p, lookback=21, skip=21),
    lambda p: momentum(p, lookback=10, skip=-1),
    lambda p: short_term_reversal(p, lookback=0),
    lambda p: realized_vol(p, window=1),
    lambda p: ma_crossover(p, fast=10, slow=10),
    lambda p: rsi(p, window=0),
    lambda p: winsorize(p, 0.5, 0.5),
    lambda p: winsorize(p, -0.1, 0.9),
])
def test_invalid_arguments_raise(bad_call):
    p = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        bad_call(p)
