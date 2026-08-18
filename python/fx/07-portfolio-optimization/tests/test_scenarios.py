"""Edge cases and crisis scenarios (documentation contract item 6).

Every case here is also documented in docs/VALIDATION.md.
"""

import numpy as np
import pandas as pd
import pytest

from fx_port import (
    carry_signal,
    empirical_cvar,
    erc_weights,
    lw_shrinkage,
    max_utility,
    min_variance_weights,
    momentum_signal,
    psd_repair,
    rank_weights,
    sample_cov,
    style_returns,
    total_log_returns,
)
from fx_port.data import make_panel


def test_single_currency_universe_degenerates_gracefully():
    panel = make_panel(seed=3, n_days=300, currencies=["EUR"])
    dec = total_log_returns(panel.spots, panel.rates)
    sig = carry_signal(panel.rates, ["EUR"]).reindex(dec.total.index)
    ret, w = style_returns(dec.total, sig)
    assert (w == 0).all().all()          # no cross-section: flat book
    assert (ret == 0).all()
    # long-only MVO still works with one asset
    mv = min_variance_weights(sample_cov(dec.total))
    assert mv["EUR"] == pytest.approx(1.0, abs=1e-12)


def test_zero_rate_differentials_kill_carry_not_pipeline():
    panel = make_panel(seed=3, n_days=200, currencies=["EUR", "JPY", "AUD"])
    rates = panel.rates.copy()
    for c in ["EUR", "JPY", "AUD"]:
        rates[c] = rates["USD"]  # peg every deposit rate to USD
    dec = total_log_returns(panel.spots, rates)
    assert (dec.carry == 0).all().all()  # carry leg exactly zero
    sig = carry_signal(rates, ["EUR", "JPY", "AUD"]).reindex(dec.total.index)
    ret, w = style_returns(dec.total, sig)
    assert (w == 0).all().all()          # degenerate signal: stand down
    assert (ret == 0).all()


def test_pegged_pair_in_universe_zero_vol_handled():
    panel = make_panel(seed=5, n_days=400, include_peg=True)
    dec = total_log_returns(panel.spots, panel.rates)
    s = sample_cov(dec.total)
    # raw covariance is near-singular; the documented remedy works:
    floored = psd_repair(s, min_eig=1e-10)
    w = min_variance_weights(floored)
    assert np.isfinite(w).all()
    # min-var loads overwhelmingly on the (near) risk-free pegged currency
    assert w["PEG"] > 0.9
    # ERC refuses a zero-vol asset with a clear message...
    tiny = s.copy()
    tiny.loc["PEG", "PEG"] = 0.0
    with pytest.raises(ValueError, match="diagonal"):
        erc_weights(tiny)
    # ...and works once the peg is dropped
    dropped = s.drop(index="PEG", columns="PEG")
    w2 = erc_weights(dropped)
    assert w2.sum() == pytest.approx(1.0, abs=1e-12)


def test_all_crash_sample_carry_loses():
    # crash every ~3 days: the carry trade must lose money and show fat tails
    panel = make_panel(seed=7, n_days=500, crash_prob=0.3)
    dec = total_log_returns(panel.spots, panel.rates)
    sig = carry_signal(panel.rates, list(panel.spots.columns))
    ret, _ = style_returns(dec.total, sig.reindex(dec.total.index))
    live = ret[ret != 0]
    assert live.mean() < 0                       # premium wiped out
    assert empirical_cvar(live, 0.95) > 0        # metrics still computable
    assert np.isfinite(live).all()


def test_gross_leverage_zero_everywhere():
    panel = make_panel(seed=1, n_days=300, currencies=["EUR", "JPY", "AUD", "NZD"])
    dec = total_log_returns(panel.spots, panel.rates)
    sig = carry_signal(panel.rates, list(panel.spots.columns)).iloc[-1]
    assert (rank_weights(sig, gross=0.0) == 0).all()
    sigma, _ = lw_shrinkage(dec.total)
    res = max_utility(dec.total.mean(), sigma, sum_to=0.0, gross_limit=0.0)
    assert (res.weights == 0).all()


def test_momentum_window_longer_than_sample():
    panel = make_panel(seed=2, n_days=100, currencies=["EUR", "JPY"])
    dec = total_log_returns(panel.spots, panel.rates)
    sig = momentum_signal(panel.spots, lookback=252, skip=21)
    assert sig.isna().all().all()  # no history: signal undefined everywhere
    ret, w = style_returns(dec.total, sig.reindex(dec.total.index))
    assert (w == 0).all().all()    # engine stands down instead of crashing
    assert (ret == 0).all()


def test_crisis_regime_correlations_flip_up():
    """Risk-on currencies decouple in calm markets, co-crash in stress."""
    calm = make_panel(seed=11, n_days=1500, crash_prob=0.0)
    stressed = make_panel(seed=11, n_days=1500, crash_prob=0.05)
    for p, label in ((calm, "calm"), (stressed, "stress")):
        p.corr = total_log_returns(p.spots, p.rates).spot[["AUD", "TRY"]].corr().iloc[0, 1]
    # crash co-movement pushes high-carry correlations up in the stressed world
    assert stressed.corr > calm.corr


def test_short_sample_covariance_raises():
    panel = make_panel(seed=1, n_days=3, currencies=["EUR", "JPY"])
    dec = total_log_returns(panel.spots, panel.rates)
    with pytest.raises(ValueError, match="at least"):
        sample_cov(dec.total.iloc[:1])
