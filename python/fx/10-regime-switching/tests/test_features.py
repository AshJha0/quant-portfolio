"""Feature-block tests: hand-checks and the PIT mutation test."""

import numpy as np
import pandas as pd
import pytest

from fx_regime import (
    FEATURE_COLUMNS,
    FeatureConfig,
    avg_pairwise_correlation,
    build_features,
    carry_basket_weights,
    expanding_standardize,
    generate_roro_panel,
    realised_vol,
)


def _tiny_returns(n=12, cols=("A", "B")):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        rng.standard_normal((n, len(cols))) * 0.01, index=idx, columns=list(cols)
    )


def test_realised_vol_hand_check():
    r = _tiny_returns(6, cols=("A",))
    out = realised_vol(r, window=3)
    x = r["A"].iloc[2:5].to_numpy()
    expected = x.std(ddof=0) * np.sqrt(252)
    assert np.isclose(out["A"].iloc[4], expected, atol=1e-12)
    assert out["A"].iloc[:2].isna().all()


def test_carry_basket_weights_membership_and_neutrality():
    rates = pd.Series(
        {"AUD": 0.05, "NZD": 0.06, "JPY": 0.0, "CHF": -0.01, "EUR": 0.02}
    )
    w = carry_basket_weights(rates, list(rates.index), n_long=2, n_short=2)
    assert np.isclose(w.sum(), 0.0)
    assert w["NZD"] == pytest.approx(0.5)
    assert w["AUD"] == pytest.approx(0.5)
    assert w["CHF"] == pytest.approx(-0.5)
    assert w["JPY"] == pytest.approx(-0.5)
    assert w["EUR"] == 0.0


def test_carry_basket_invalid_sizes_raise():
    rates = pd.Series({"A": 0.01, "B": 0.02})
    with pytest.raises(ValueError):
        carry_basket_weights(rates, ["A", "B"], n_long=2, n_short=2)
    with pytest.raises(ValueError):
        carry_basket_weights(rates, ["A", "B"], n_long=0, n_short=1)


def test_avg_pairwise_correlation_hand_check():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = np.arange(10, dtype=float)
    df = pd.DataFrame({"A": a, "B": 2 * a, "C": -a}, index=idx)
    out = avg_pairwise_correlation(df, window=5)
    # corr(A,B)=1, corr(A,C)=-1, corr(B,C)=-1 -> mean = -1/3
    assert np.isclose(out.iloc[-1], -1.0 / 3.0, atol=1e-10)
    assert out.iloc[:3].isna().all()


def test_avg_pairwise_correlation_pegged_currency_handled():
    idx = pd.bdate_range("2020-01-01", periods=30)
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "A": rng.standard_normal(30),
            "B": rng.standard_normal(30),
            "PEG": np.zeros(30),  # pegged: zero vol
        },
        index=idx,
    )
    out = avg_pairwise_correlation(df, window=10)
    valid = out.iloc[9:]
    assert np.isfinite(valid).all()
    # only the (A, B) pair contributes
    expected = df["A"].rolling(10).corr(df["B"]).iloc[-1]
    assert np.isclose(out.iloc[-1], expected, atol=1e-12)


def test_expanding_standardize_matches_manual():
    df = _tiny_returns(50)
    z = expanding_standardize(df, min_periods=10)
    t = 30
    x = df["A"].iloc[: t + 1]
    expected = (x.iloc[-1] - x.mean()) / x.std(ddof=0)
    assert np.isclose(z["A"].iloc[t], expected, atol=1e-12)
    assert z["A"].iloc[:9].isna().all()


def test_expanding_standardize_pit_mutation():
    """CRITICAL: perturbing the future must not change the past."""
    df = _tiny_returns(100)
    z1 = expanding_standardize(df, min_periods=10)
    df2 = df.copy()
    df2.iloc[60:] += 5.0  # violent future shock
    z2 = expanding_standardize(df2, min_periods=10)
    pd.testing.assert_frame_equal(z1.iloc[:60], z2.iloc[:60])
    assert not np.allclose(
        z1.iloc[60:].to_numpy(), z2.iloc[60:].to_numpy()
    )


def test_expanding_standardize_zero_variance_column():
    idx = pd.bdate_range("2020-01-01", periods=20)
    df = pd.DataFrame({"A": np.ones(20)}, index=idx)
    z = expanding_standardize(df, min_periods=5)
    assert (z["A"].iloc[4:] == 0.0).all()


def test_build_features_columns_and_no_nans(panel2):
    feats = build_features(panel2.returns, panel2.deposit_rates)
    assert tuple(feats.columns) == FEATURE_COLUMNS
    assert not feats.isna().any().any()
    assert len(feats) > 900


def test_build_features_pit_mutation(panel2):
    """CRITICAL: features before t unchanged when returns after t change."""
    feats1 = build_features(panel2.returns, panel2.deposit_rates)
    cut = 700
    cut_date = panel2.returns.index[cut]
    rets2 = panel2.returns.copy()
    rets2.iloc[cut:] *= -3.0
    feats2 = build_features(rets2, panel2.deposit_rates)
    common = feats1.index[feats1.index < cut_date]
    pd.testing.assert_frame_equal(
        feats1.loc[common], feats2.loc[common]
    )


def test_features_separate_regimes():
    """avg_vol and usd_corr must be higher in risk_off than risk_on
    (long sample so per-state feature means are statistically clean)."""
    panel = generate_roro_panel(4000, n_states=2, seed=0)
    feats = build_features(
        panel.returns, panel.deposit_rates, standardize=False
    )
    states = pd.Series(panel.states, index=panel.returns.index).reindex(
        feats.index
    )
    on, off = states == 0, states == 1
    assert feats.loc[off, "avg_vol"].mean() > feats.loc[on, "avg_vol"].mean()
    assert feats.loc[off, "usd_corr"].mean() > feats.loc[on, "usd_corr"].mean()
    assert feats.loc[off, "haven_rs"].mean() > feats.loc[on, "haven_rs"].mean()
    assert feats.loc[off, "carry_ret"].mean() < feats.loc[on, "carry_ret"].mean()


def test_fwd_ts_positive_for_carry_longs(panel2):
    """Forward-point proxy: carry longs trade at a forward discount > 0."""
    feats = build_features(
        panel2.returns, panel2.deposit_rates, standardize=False
    )
    assert (feats["fwd_ts"] > 0).all()
    # magnitude ~ average carry-long rate differential (a few percent)
    assert 0.01 < feats["fwd_ts"].mean() < 0.15


def test_short_series_raises():
    panel = generate_roro_panel(60, n_states=2, seed=0)
    with pytest.raises(ValueError, match="too short"):
        build_features(panel.returns, panel.deposit_rates)


def test_missing_usd_rate_raises(panel2):
    with pytest.raises(ValueError, match="USD"):
        build_features(
            panel2.returns, panel2.deposit_rates.drop(columns=["USD"])
        )
