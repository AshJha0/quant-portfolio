"""Tests for the synthetic FX market generator (structure, not just shape)."""

import numpy as np
import pandas as pd
import pytest

from fx_port import (
    carry_signal,
    one_factor_cov,
    sharpe_ratio,
    skewness,
    style_returns,
    total_log_returns,
)
from fx_port.data import make_equity_portfolio, make_panel


@pytest.fixture(scope="module")
def panel():
    return make_panel(seed=0, n_days=3000)


def test_seed_determinism():
    a = make_panel(seed=42, n_days=100)
    b = make_panel(seed=42, n_days=100)
    pd.testing.assert_frame_equal(a.spots, b.spots)
    pd.testing.assert_frame_equal(a.rates, b.rates)
    pd.testing.assert_frame_equal(a.ppp, b.ppp)
    c = make_panel(seed=43, n_days=100)
    assert not a.spots.equals(c.spots)


def test_carry_style_high_sharpe_negative_skew(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    sig = carry_signal(panel.rates, list(panel.spots.columns))
    ret, _ = style_returns(dec.total, sig.reindex(dec.total.index))
    live = ret[ret != 0]
    assert sharpe_ratio(live) > 0.3          # carry earns its premium...
    assert skewness(live) < -0.5             # ...with crash-driven negative skew


def test_carry_crash_days_hit_high_yielders(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    crash = panel.crash_days.reindex(dec.spot.index).fillna(False)
    assert crash.sum() >= 3  # some crash days in ~12 years
    on = dec.spot[crash]
    # high-carry currencies fall much harder than the funding currency on
    # crash days (the carry-crash mechanism)
    assert on["TRY"].mean() < on["JPY"].mean() - 0.01
    assert on["BRL"].mean() < 0


def test_risk_factor_loadings_signs(panel):
    dec = total_log_returns(panel.spots, panel.rates)
    model = one_factor_cov(dec.spot)
    assert model.loadings["AUD"] > 0
    assert model.loadings["NZD"] > 0
    assert model.loadings["MXN"] > 0
    assert model.loadings["JPY"] < 0
    assert model.loadings["CHF"] < 0


def test_rate_differentials_persistent(panel):
    diff = panel.rates.sub(panel.rates["USD"], axis=0)
    assert diff["TRY"].mean() > 0.05      # EM stays high-yield
    assert diff["JPY"].mean() < -0.01     # JPY stays a funding currency
    # persistence: yearly mean differentials rarely change sign
    yearly = diff["TRY"].resample("YE").mean()
    assert (yearly > 0).all()


def test_rates_non_negative_and_spots_positive(panel):
    assert (panel.rates >= 0).all().all()
    assert (panel.spots > 0).all().all()
    assert list(panel.ppp.columns) == list(panel.spots.columns)


def test_spots_track_ppp_loosely(panel):
    gap = np.log(panel.spots / panel.ppp)
    # stationary-ish misvaluation: bounded, not a divergent random walk
    # (crash drift can push high-carry currencies persistently below PPP,
    # so the band is wide — see METHODOLOGY assumptions register)
    assert gap.abs().to_numpy().max() < 2.0
    # G10 misvaluations stay in a tighter economic range
    assert gap[["EUR", "GBP", "CHF"]].abs().to_numpy().max() < 1.0


def test_pegged_currency_near_zero_vol():
    p = make_panel(seed=4, n_days=400, include_peg=True)
    ret = np.log(p.spots["PEG"]).diff().dropna()
    assert ret.std() < 1e-5
    assert np.allclose(p.rates["PEG"], p.rates["USD"])  # zero differential


def test_unknown_currency_raises():
    with pytest.raises(ValueError, match="unknown"):
        make_panel(currencies=["EUR", "XXX"])


def test_too_short_panel_raises():
    with pytest.raises(ValueError, match="n_days"):
        make_panel(n_days=1)
    with pytest.raises(ValueError, match="n_days"):
        make_equity_portfolio(n_days=0)


def test_equity_portfolio_identity():
    m = make_equity_portfolio(seed=8, n_days=300)
    rebuilt = (m.local_returns * pd.Series(
        [0.40, 0.20, 0.12, 0.10, 0.08, 0.10], index=m.local_returns.columns
    )).sum(axis=1) + (m.fx_returns * m.exposures).sum(axis=1)
    assert np.max(np.abs(m.unhedged_returns - rebuilt)) < 1e-15
    assert list(m.fx_returns.columns) == list(m.exposures.index)
    assert (m.exposures > 0).all()


def test_equity_fx_safe_haven_correlations():
    m = make_equity_portfolio(seed=8, n_days=2500)
    eq = m.local_returns["US"]
    assert m.fx_returns["JPY"].corr(eq) < 0   # safe haven
    assert m.fx_returns["AUD"].corr(eq) > 0   # risk-on
