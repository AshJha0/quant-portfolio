"""Signal tests: IC vs scipy, planted vs noise alpha, decile monotonicity,
banded rebalancing, combination utilities."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eq_algo import (apply_rebalance_band, combine_equal_weight,
                     combine_ic_weighted, cs_zscore, decile_portfolios,
                     forward_returns, freeze_signal, generate_daily_panel,
                     information_coefficient, ic_summary, momentum,
                     signal_decay, turnover)


@pytest.fixture(scope="module")
def big_panel():
    return generate_daily_panel(n_stocks=100, n_days=900, seed=5)


def test_ic_matches_scipy_spearmanr():
    rng = np.random.default_rng(1)
    sig = pd.DataFrame(rng.standard_normal((15, 30)))
    fwd = pd.DataFrame(rng.standard_normal((15, 30)))
    sig[sig > 1.5] = np.nan  # missing data handled identically
    ic = information_coefficient(sig, fwd)
    for t in sig.index:
        mask = sig.loc[t].notna() & fwd.loc[t].notna()
        rho, _ = stats.spearmanr(sig.loc[t][mask], fwd.loc[t][mask])
        assert ic.loc[t] == pytest.approx(rho, abs=1e-12)


def test_ic_handles_ties_like_scipy():
    sig = pd.DataFrame([[1.0, 1.0, 2.0, 3.0, 3.0, 4.0]])
    fwd = pd.DataFrame([[0.1, 0.4, 0.2, 0.5, 0.3, 0.9]])
    rho, _ = stats.spearmanr(sig.iloc[0], fwd.iloc[0])
    assert information_coefficient(sig, fwd).iloc[0] == pytest.approx(rho, abs=1e-12)


def test_planted_alpha_has_significant_ic(big_panel):
    mom = momentum(big_panel.prices, 252, 21)
    fwd = forward_returns(big_panel.prices, 1)
    s = ic_summary(information_coefficient(mom, fwd))
    assert 0.01 < s["mean_ic"] < 0.08          # planted IC ~ 0.04
    assert s["tstat_nw"] > 2.0
    assert s["tstat_naive"] > 2.0


def test_noise_feature_has_no_ic(big_panel):
    rng = np.random.default_rng(99)
    noise = pd.DataFrame(rng.standard_normal(big_panel.prices.shape),
                         index=big_panel.prices.index,
                         columns=big_panel.prices.columns)
    fwd = forward_returns(big_panel.prices, 1)
    s = ic_summary(information_coefficient(noise, fwd))
    assert abs(s["tstat_naive"]) < 2.0
    assert abs(s["mean_ic"]) < 0.01


def test_decile_portfolios_monotone_for_planted_alpha(big_panel):
    mom = cs_zscore(momentum(big_panel.prices, 252, 21))
    fwd = forward_returns(big_panel.prices, 1)
    dec = decile_portfolios(mom, fwd, n_quantiles=10)
    means = dec.mean()
    assert means["Q10"] > means["Q1"]
    assert means["LS"] > 0
    # monotone rank correlation across the 10 buckets
    qs = means[[f"Q{i}" for i in range(1, 11)]].to_numpy()
    rho, _ = stats.spearmanr(np.arange(10), qs)
    assert rho > 0.85


def test_decile_portfolios_hand_computed():
    sig = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], columns=list("ABCD"))
    fwd = pd.DataFrame([[0.01, 0.02, 0.05, 0.07]], columns=list("ABCD"))
    dec = decile_portfolios(sig, fwd, n_quantiles=2)
    assert dec["Q1"].iloc[0] == pytest.approx(0.015, abs=1e-12)
    assert dec["Q2"].iloc[0] == pytest.approx(0.06, abs=1e-12)
    assert dec["LS"].iloc[0] == pytest.approx(0.045, abs=1e-12)


def test_signal_decay_table(big_panel):
    mom = momentum(big_panel.prices, 252, 21)
    tab = signal_decay(mom, big_panel.prices, horizons=[1, 5, 10])
    assert list(tab.index) == [1, 5, 10]
    ic1 = information_coefficient(mom, forward_returns(big_panel.prices, 1)).mean()
    assert tab.loc[1, "mean_ic"] == pytest.approx(ic1, rel=1e-10)
    assert (tab["n_obs"] > 0).all()


def test_forward_returns_alignment():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    f = forward_returns(p, 1)
    assert f["A"].iloc[0] == pytest.approx(0.10, abs=1e-12)
    assert f["A"].iloc[1] == pytest.approx(-0.10, abs=1e-12)
    assert np.isnan(f["A"].iloc[2])


def test_band_rebalancing_reduces_turnover():
    rng = np.random.default_rng(4)
    target = pd.DataFrame(rng.standard_normal((300, 20)) * 0.02)
    naive = apply_rebalance_band(target, 0.0)
    banded = apply_rebalance_band(target, 0.01)
    pd.testing.assert_frame_equal(naive, target)  # band 0 == naive
    assert turnover(banded).sum() < turnover(naive).sum()
    # held book never drifts more than the band away from target
    assert (banded - target).abs().to_numpy().max() <= 0.01 + 1e-12


def test_band_rebalancing_hand_computed():
    target = pd.DataFrame({"A": [0.10, 0.105, 0.13]})
    held = apply_rebalance_band(target, 0.01)
    # day0: trade to 0.10; day1: |0.105-0.10|<=band -> hold; day2: trade
    np.testing.assert_allclose(held["A"].to_numpy(), [0.10, 0.10, 0.13])


def test_freeze_signal_hand_computed():
    sig = pd.DataFrame({"A": [1.0, 1.05, 1.4, np.nan, 1.45],
                        "B": [0.0, 0.5, 0.6, 0.7, 0.75]})
    frozen = freeze_signal(sig, band=0.2)
    # A: init 1.0; 1.05 within band -> 1.0; 1.4 refresh; NaN -> NaN (leaves
    # universe, keeps memory); 1.45 within band of 1.4 -> 1.4
    np.testing.assert_allclose(frozen["A"].to_numpy(),
                               [1.0, 1.0, 1.4, np.nan, 1.4])
    # B: init 0.0; 0.5 refresh; 0.6/0.7 within band of 0.5; 0.75 refreshes
    np.testing.assert_allclose(frozen["B"].to_numpy(), [0.0, 0.5, 0.5, 0.5, 0.75])
    pd.testing.assert_frame_equal(freeze_signal(sig, 0.0), sig)  # band 0 = identity
    with pytest.raises(ValueError):
        freeze_signal(sig, -0.1)


def test_combine_equal_weight_identical_features():
    rng = np.random.default_rng(7)
    f = pd.DataFrame(rng.standard_normal((10, 8)))
    combo = combine_equal_weight({"a": f, "b": f.copy()})
    pd.testing.assert_frame_equal(combo, cs_zscore(f))


def test_combine_ic_weighted_tilts_to_planted(big_panel):
    prices = big_panel.prices
    fwd = forward_returns(prices, 1)
    planted = momentum(prices, 252, 21)
    rng = np.random.default_rng(13)
    noise = pd.DataFrame(rng.standard_normal(prices.shape),
                         index=prices.index, columns=prices.columns)
    combo = combine_ic_weighted({"planted": planted, "noise": noise}, fwd,
                                min_history=60)
    tail = slice(-200, None)
    z_p = cs_zscore(planted).iloc[tail].stack().rename("p")
    z_n = cs_zscore(noise).iloc[tail].stack().rename("n")
    c = combo.iloc[tail].stack().rename("c")
    df = pd.concat([c, z_p, z_n], axis=1).dropna()
    assert df["c"].corr(df["p"]) > 0.9          # dominated by the real signal
    assert abs(df["c"].corr(df["n"])) < 0.5


def test_turnover_hand_computed():
    w = pd.DataFrame({"A": [0.5, 0.2], "B": [-0.5, -0.6]})
    to = turnover(w)
    assert to.iloc[0] == pytest.approx(1.0, abs=1e-12)   # from flat
    assert to.iloc[1] == pytest.approx(0.4, abs=1e-12)


@pytest.mark.parametrize("bad_call", [
    lambda: forward_returns(pd.DataFrame({"A": [1.0, 2.0]}), 0),
    lambda: decile_portfolios(pd.DataFrame([[1.0, 2.0]]),
                              pd.DataFrame([[0.1, 0.2]]), n_quantiles=1),
    lambda: apply_rebalance_band(pd.DataFrame([[0.1]]), -0.1),
    lambda: combine_equal_weight({}),
])
def test_invalid_arguments_raise(bad_call):
    with pytest.raises(ValueError):
        bad_call()
