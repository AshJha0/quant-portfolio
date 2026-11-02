"""FX Euler attribution: currency mapping, premium-adjusted option exposure, grouping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fx_var.fx_attribution import (
    currency_exposures,
    fx_option_exposure,
    fx_parametric_attribution,
    group_components,
)
from fx_var.gk import gk_price
from fx_var.risk_attribution import parametric_attribution, simulation_attribution
from scipy.stats import norm

CCY = ("EUR", "JPY", "GBP")
X = {"EUR": 1.10, "JPY": 1.0 / 150.0, "GBP": 1.27}  # USD per unit
VOL = np.array([0.006, 0.007, 0.0065])  # daily
CORR = np.array([[1.0, 0.3, 0.7], [0.3, 1.0, 0.25], [0.7, 0.25, 1.0]])
COV = np.outer(VOL, VOL) * CORR


def test_eur_jpy_book_by_currency():
    """Worked example: long 10m EUR, short 1.5bn JPY, USD reporting, 99 % 1-day VaR."""
    e = currency_exposures({"EUR": 10e6, "JPY": -1.5e9}, X, CCY)
    np.testing.assert_allclose(e, [11e6, -10e6, 0.0], rtol=1e-15)
    a = fx_parametric_attribution(dict(zip(CCY, e, strict=True)), COV, CCY, 0.01)
    sig = np.sqrt(e @ COV @ e)
    assert a.total == pytest.approx(norm.ppf(0.99) * sig, rel=1e-14)
    comp = a.component()
    assert sum(comp.values()) == pytest.approx(a.total, rel=1e-13)
    assert comp["GBP"] == 0.0
    # EUR long and JPY short are positively correlated -> a partial hedge, both still add risk
    assert comp["EUR"] > 0.0 and comp["JPY"] > 0.0


def test_grouping_is_additive():
    e = {"EUR": 5e6, "JPY": -3e6, "GBP": 4e6}
    a = fx_parametric_attribution(e, COV, CCY, 0.025, measure="es")
    g = group_components(a, {"Europe": ["EUR", "GBP"], "Asia": ["JPY"]})
    assert g["Europe"] + g["Asia"] == pytest.approx(a.total, rel=1e-13)
    with pytest.raises(ValueError):
        group_components(a, {"Europe": ["EUR"], "Asia": ["JPY"]})
    with pytest.raises(ValueError):
        group_components(a, {"A": ["EUR", "GBP", "JPY", "EUR"]})


def test_matches_equity_twin_on_same_exposures():
    src = Path(__file__).resolve().parents[3] / "equity" / "03-var-es-engine" / "src"
    if not src.is_dir():
        pytest.skip("equity package not in this checkout")
    sys.path.insert(0, str(src))
    eq = pytest.importorskip("eq_var.risk_attribution")
    e = np.array([5e6, -3e6, 4e6])
    a = fx_parametric_attribution(dict(zip(CCY, e, strict=True)), COV, CCY, 0.01)
    b = eq.parametric_attribution(e, COV, 0.01)
    np.testing.assert_allclose(a.result.component, b.component, rtol=1e-15)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_option_exposure_quote_reporting_is_fd_of_value(kind):
    N, S, K, T, rd, rf, v = 1e6, 1.10, 1.12, 0.5, 0.045, 0.035, 0.08
    e = fx_option_exposure(N, S, K, T, rd, rf, v, kind, "quote")
    h = 1e-6
    up = N * gk_price(S * np.exp(h), K, T, rd, rf, v, kind)
    dn = N * gk_price(S * np.exp(-h), K, T, rd, rf, v, kind)
    assert e == pytest.approx(float((up - dn) / (2 * h)), rel=1e-7)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_option_exposure_base_reporting_is_premium_adjusted(kind):
    """USD book, USDJPY option: value in USD is V/S; its ln S derivative is the pa delta."""
    N, S, K, T, rd, rf, v = 1e6, 150.0, 148.0, 0.5, 0.002, 0.045, 0.10
    e = fx_option_exposure(N, S, K, T, rd, rf, v, kind, "base")
    h = 1e-6

    def val(s):
        return N * gk_price(s, K, T, rd, rf, v, kind) / s

    fd = float((val(S * np.exp(h)) - val(S * np.exp(-h))) / (2 * h))
    assert e == pytest.approx(fd, rel=1e-7)
    # pa closed form: Delta_pa = w e^{-r_f T} (K/F) N(w d2)
    w = 1 if kind == "call" else -1
    F = S * np.exp((rd - rf) * T)
    d2 = (np.log(F / K) - 0.5 * v * v * T) / (v * np.sqrt(T))
    assert e == pytest.approx(N * w * np.exp(-rf * T) * (K / F) * norm.cdf(w * d2), rel=1e-12)


def test_option_book_delta_mapping_vs_full_revaluation():
    """Delta-mapped simulation reproduces the parametric split; full revaluation of a
    long-gamma book has a lower ES than the linear map (convexity), never higher."""
    rng = np.random.default_rng(5)
    S = np.array([1.10, 150.0])
    vol1d = np.array([0.006, 0.007])
    cov = np.outer(vol1d, vol1d) * np.array([[1.0, 0.3], [0.3, 1.0]])
    # EURUSD call (USD reporting = quote) and USDJPY put (USD reporting = base)
    e_eur = fx_option_exposure(2e6, S[0], 1.10, 0.5, 0.045, 0.035, 0.08, "call", "quote")
    e_jpy = fx_option_exposure(1e6, S[1], 150.0, 0.5, 0.002, 0.045, 0.10, "put", "base")
    # factor for USDJPY is ln(USD/JPY) itself here, so the label is the pair
    a = fx_parametric_attribution({"EURUSD": e_eur, "USDJPY": e_jpy}, cov,
                                  ("EURUSD", "USDJPY"), 0.025, measure="es")
    r = rng.multivariate_normal(np.zeros(2), cov, size=400_000)
    v0e = 2e6 * gk_price(S[0], 1.10, 0.5, 0.045, 0.035, 0.08, "call")
    v0j = 1e6 * gk_price(S[1], 150.0, 0.5, 0.002, 0.045, 0.10, "put") / S[1]
    S1 = S * np.exp(r)
    pe = 2e6 * gk_price(S1[:, 0], 1.10, 0.5, 0.045, 0.035, 0.08, "call") - v0e
    pj = 1e6 * gk_price(S1[:, 1], 150.0, 0.5, 0.002, 0.045, 0.10, "put") / S1[:, 1] - v0j
    lin = simulation_attribution(r * np.array([e_eur, e_jpy]), 0.025, measure="es")
    assert lin.total == pytest.approx(a.total, rel=0.02)
    np.testing.assert_allclose(lin.component, a.result.component, rtol=0.05)
    full = simulation_attribution(np.column_stack([pe, pj]), 0.025, measure="es")
    assert 0.8 * a.total < full.total < a.total
    assert full.component.sum() == pytest.approx(full.total, rel=1e-12)


def test_guards():
    with pytest.raises(ValueError, match="no risk factor"):
        currency_exposures({"USD": 1.0}, X, CCY)
    with pytest.raises(ValueError, match="no rate"):
        currency_exposures({"EUR": 1.0}, {"EUR": 1.1}, CCY)
    with pytest.raises(ValueError):
        currency_exposures({"EUR": float("nan")}, X, CCY)
    with pytest.raises(ValueError):
        currency_exposures({"EUR": 1.0}, {**X, "JPY": 0.0}, CCY)
    with pytest.raises(ValueError, match="reporting"):
        fx_option_exposure(1.0, 1.1, 1.1, 0.5, 0.0, 0.0, 0.1, "call", "foreign")
    with pytest.raises(ValueError, match="unknown labels"):
        fx_parametric_attribution({"CHF": 1.0}, COV, CCY)
    with pytest.raises(ValueError):
        fx_parametric_attribution({"EUR": 0.0}, COV, CCY)  # zero-variance book
    assert parametric_attribution  # re-exported building block stays importable
