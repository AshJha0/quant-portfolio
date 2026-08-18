"""Tests for total-return construction and style signals."""

import numpy as np
import pandas as pd
import pytest

from fx_port import (
    carry_log_returns,
    carry_signal,
    momentum_signal,
    rank_weights,
    shrunk_means,
    spot_log_returns,
    style_returns,
    total_log_returns,
    value_signal,
)
from fx_port.data import make_panel

DT = 1.0 / 252


@pytest.fixture(scope="module")
def panel():
    return make_panel(seed=11, n_days=800)


def test_total_equals_spot_plus_carry_exact(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    err = (dec.total - (dec.spot + dec.carry)).abs().to_numpy().max()
    assert err == 0.0  # exact additive identity by construction


def test_spot_returns_match_log_diff(panel):
    sr = spot_log_returns(panel.spots)
    manual = np.log(panel.spots.iloc[5]["EUR"] / panel.spots.iloc[4]["EUR"])
    assert sr.iloc[4]["EUR"] == pytest.approx(manual, abs=1e-15)


def test_carry_uses_previous_day_rates():
    idx = pd.bdate_range("2020-01-01", periods=4)
    rates = pd.DataFrame({"AUD": [0.04, 0.08, 0.08, 0.08], "USD": 0.02}, index=idx)
    carry = carry_log_returns(rates, ["AUD"], base="USD")
    # row for day 1 (second date) uses day-0 rates: (0.04-0.02)*dt
    assert carry.iloc[0]["AUD"] == pytest.approx(0.02 * DT, abs=1e-15)
    # day 2 uses day-1 rates: (0.08-0.02)*dt — the jump arrives with a lag
    assert carry.iloc[1]["AUD"] == pytest.approx(0.06 * DT, abs=1e-15)


def test_spot_returns_reject_nonpositive(panel):
    bad = panel.spots.copy()
    bad.iloc[3, 0] = -1.0
    with pytest.raises(ValueError, match="positive"):
        spot_log_returns(bad)


def test_carry_missing_base_raises(panel):
    with pytest.raises(ValueError, match="base"):
        carry_log_returns(panel.rates.drop(columns="USD"), ["EUR"])


def test_total_requires_rate_coverage(panel):
    with pytest.raises(ValueError, match="cover"):
        total_log_returns(panel.spots, panel.rates.iloc[100:])


def test_carry_signal_matches_rate_differential(panel):
    sig = carry_signal(panel.rates, ["AUD", "JPY"])
    expected = panel.rates["AUD"] - panel.rates["USD"]
    assert np.allclose(sig["AUD"], expected)
    # JPY differential is deeply negative (funding currency)
    assert sig["JPY"].mean() < -0.01


def test_carry_ranks_match_rate_differentials(panel):
    sig = carry_signal(panel.rates, list(panel.spots.columns)).iloc[-1]
    w = rank_weights(sig, gross=2.0)
    # weight ordering must equal signal ordering
    assert (w[sig.sort_values().index].diff().dropna() > 0).all()
    assert w.idxmax() == sig.idxmax()
    assert w.idxmin() == sig.idxmin()


def test_momentum_lookback_correct():
    idx = pd.bdate_range("2020-01-01", periods=300)
    g = 0.001
    spots = pd.DataFrame({"EUR": np.exp(g * np.arange(300))}, index=idx)
    mom = momentum_signal(spots, lookback=252, skip=21)
    # log(S[t-21]/S[t-252]) = g * (252 - 21)
    assert mom.iloc[-1]["EUR"] == pytest.approx(g * 231, rel=1e-12)
    assert mom.iloc[:251].isna().all().all()  # undefined inside the window


def test_momentum_no_lookahead():
    idx = pd.bdate_range("2020-01-01", periods=320)
    rng = np.random.default_rng(0)
    spots = pd.DataFrame(
        {"EUR": np.exp(np.cumsum(0.005 * rng.standard_normal(320)))}, index=idx
    )
    mom = momentum_signal(spots)
    bumped = spots.copy()
    bumped.iloc[300:] *= 10.0  # violent future move
    mom2 = momentum_signal(bumped)
    pd.testing.assert_frame_equal(mom.iloc[:280], mom2.iloc[:280])


def test_momentum_invalid_params(panel):
    with pytest.raises(ValueError, match="lookback"):
        momentum_signal(panel.spots, lookback=10, skip=21)


def test_value_sign_vs_ppp_gap():
    idx = pd.bdate_range("2020-01-01", periods=3)
    spots = pd.DataFrame({"EUR": [1.0, 1.0, 1.0]}, index=idx)
    ppp = pd.DataFrame({"EUR": [1.2, 1.0, 0.8]}, index=idx)
    val = value_signal(spots, ppp)
    assert val.iloc[0]["EUR"] > 0  # spot below PPP: undervalued, long
    assert val.iloc[1]["EUR"] == pytest.approx(0.0, abs=1e-15)
    assert val.iloc[2]["EUR"] < 0  # overvalued, short


def test_value_mismatched_columns_raise(panel):
    with pytest.raises(ValueError, match="columns"):
        value_signal(panel.spots, panel.ppp[panel.ppp.columns[::-1]])


def test_rank_weights_dollar_neutral_and_gross():
    sig = pd.Series([3.0, -1.0, 0.5, 2.0, -2.0], index=list("ABCDE"))
    w = rank_weights(sig, gross=1.6)
    assert w.sum() == pytest.approx(0.0, abs=1e-14)
    assert w.abs().sum() == pytest.approx(1.6, rel=1e-14)


def test_rank_weights_degenerate_all_equal():
    w = rank_weights(pd.Series([1.0, 1.0, 1.0]), gross=2.0)
    assert (w == 0).all()


def test_rank_weights_gross_zero():
    w = rank_weights(pd.Series([1.0, 2.0, 3.0]), gross=0.0)
    assert (w == 0).all()


def test_rank_weights_negative_gross_raises():
    with pytest.raises(ValueError, match="gross"):
        rank_weights(pd.Series([1.0, 2.0]), gross=-1.0)


def test_rank_weights_single_currency():
    w = rank_weights(pd.Series([5.0], index=["EUR"]), gross=2.0)
    assert (w == 0).all()  # cannot be long-short with one name


def test_rank_weights_nan_excluded():
    sig = pd.Series([1.0, np.nan, 3.0], index=list("ABC"))
    w = rank_weights(sig, gross=2.0)
    assert w["B"] == 0.0
    assert w.abs().sum() == pytest.approx(2.0)


def test_style_returns_apply_one_day_lag(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    sig = carry_signal(panel.rates, list(panel.spots.columns)).reindex(dec.total.index)
    ret, applied = style_returns(dec.total, sig, gross=2.0)
    # weights applied on day t must equal rank weights of signal at t-1
    t = 100
    expected = rank_weights(sig.iloc[t - 1], gross=2.0)
    assert np.allclose(applied.iloc[t], expected)
    assert ret.iloc[t] == pytest.approx(
        float((expected * dec.total.iloc[t]).sum()), abs=1e-15
    )


def test_style_returns_columns_mismatch(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    with pytest.raises(ValueError, match="columns"):
        style_returns(dec.total, dec.total[dec.total.columns[::-1]])


def test_zero_rate_differentials_carry_degenerate():
    idx = pd.bdate_range("2020-01-01", periods=50)
    rates = pd.DataFrame(0.02, index=idx, columns=["EUR", "JPY", "USD"])
    sig = carry_signal(rates, ["EUR", "JPY"])
    assert (sig == 0).all().all()
    w = rank_weights(sig.iloc[-1])
    assert (w == 0).all()  # degenerate cross-section: stand down, not crash


def test_shrunk_means_intensity_bounds(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    mu, lam = shrunk_means(dec.total)
    assert 0.0 <= lam <= 1.0
    sample = dec.total.mean()
    grand = sample.mean()
    # shrunk means lie between sample means and the grand mean
    assert ((mu - grand).abs() <= (sample - grand).abs() + 1e-18).all()


def test_shrunk_means_full_shrink_equals_grand_mean(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    mu, lam = shrunk_means(dec.total, intensity=1.0)
    assert lam == 1.0
    assert np.allclose(mu, dec.total.mean().mean())


def test_shrunk_means_invalid_intensity(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    with pytest.raises(ValueError, match="intensity"):
        shrunk_means(dec.total, intensity=1.5)


def test_shrunk_means_few_assets_no_shrink():
    idx = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.standard_normal((100, 2)) * 0.01, index=idx, columns=["A", "B"])
    mu, lam = shrunk_means(r)
    assert lam == 0.0  # James-Stein needs K > 3
    assert np.allclose(mu, r.mean())
