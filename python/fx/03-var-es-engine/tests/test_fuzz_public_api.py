"""Fuzz every public entry point with non-finite and invalid inputs.

``CASES`` maps each public callable to one or more injection sites: a
function ``f(x)`` that calls the entry point with ``x`` in one argument and
valid values everywhere else, plus a valid ``x``.  The test checks that the
valid call succeeds (so the error is caused by the injected value) and
that NaN, +inf and -inf all raise ``ValueError``.

``test_registry_covers_public_api`` fails when a new public callable is
added to ``fx_var.__all__`` without a fuzz entry or a stated exemption.
"""

from __future__ import annotations

import warnings

import fx_var as fv
import numpy as np
import pandas as pd
import pytest
from fx_var.data.synthetic import demo_market
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

MKT = demo_market()
EUR_BOOK = fv.Book([fv.Spot("EURUSD", 1e6)])
RETS = pd.DataFrame({"FX:EUR": np.random.default_rng(0).normal(0, 0.006, 300)})
COV1 = pd.DataFrame([[3.6e-5]], index=["FX:EUR"], columns=["FX:EUR"])
PNL = np.random.default_rng(1).normal(0, 1e5, 300)
W2 = pd.Series({"a": 1e6, "b": -5e5})
COV2 = pd.DataFrame([[4e-5, 1e-5], [1e-5, 3e-5]], index=["a", "b"], columns=["a", "b"])
BAD = [np.nan, np.inf, -np.inf]


def _rets_with(x):
    r = RETS.copy()
    r.iloc[100, 0] = x
    return r


def _arr_with(a, x, i=17):
    a = np.array(a, dtype=float).copy()
    a[i] = x
    return a


def _cov_with(x):
    c = COV2.copy()
    c.iloc[0, 1] = c.iloc[1, 0] = x
    return c


def _rolling(x):
    return fv.rolling_backtest(EUR_BOOK, MKT, _rets_with(x),
                               lambda b, m, w: fv.historical_var(b, m, w, 0.05, min_obs=30).var,
                               window=250)


#: name -> list of (label, f(x), valid x)
CASES: dict[str, list[tuple[str, object, object]]] = {
    "Market": [("spot", lambda x: fv.Market({"EUR": x}, {"USD": 0.05}), 1.08),
               ("rate", lambda x: fv.Market({"EUR": 1.08}, {"USD": x}), 0.05),
               ("vol", lambda x: fv.Market({"EUR": 1.08}, {"USD": 0.05}, {"EURUSD": x}), 0.1)],
    "Book": [("shock", lambda x: EUR_BOOK.pnl(MKT, {"FX:EUR": x}), 0.01)],
    "Cash": [("amount", lambda x: fv.Cash("EUR", x), 1e6)],
    "Spot": [("notional", lambda x: fv.Spot("EURUSD", x), 1e6)],
    "Forward": [("notional", lambda x: fv.Forward("EURUSD", x, 0.5), 1e6),
                ("expiry", lambda x: fv.Forward("EURUSD", 1e6, x), 0.5)],
    "Option": [("strike", lambda x: fv.Option("EURUSD", 1e6, x, 0.25), 1.1),
               ("expiry", lambda x: fv.Option("EURUSD", 1e6, 1.1, x), 0.25),
               ("notional", lambda x: fv.Option("EURUSD", x, 1.1, 0.25), 1e6)],
    "split_pair": [("pair", lambda x: fv.split_pair(x), "EURUSD")],
    "fx_factor": [("ccy", lambda x: fv.fx_factor(x), "EUR")],
    "ir_factor": [("ccy", lambda x: fv.ir_factor(x), "EUR")],
    "vol_factor": [("pair", lambda x: fv.vol_factor(x), "EURUSD")],
    "gk_price": [("spot", lambda x: fv.gk_price(x, 1.1, 0.5, 0.05, 0.03, 0.1), 1.08),
                 ("rate", lambda x: fv.gk_price(1.08, 1.1, 0.5, x, 0.03, 0.1), 0.05)],
    "gk_delta": [("spot", lambda x: fv.gk_delta(x, 1.1, 0.5, 0.05, 0.03, 0.1), 1.08)],
    "gk_gamma": [("spot", lambda x: fv.gk_gamma(x, 1.1, 0.5, 0.05, 0.03, 0.1), 1.08)],
    "gk_vega": [("spot", lambda x: fv.gk_vega(x, 1.1, 0.5, 0.05, 0.03, 0.1), 1.08)],
    "historical_var": [("returns", lambda x: fv.historical_var(EUR_BOOK, MKT, _rets_with(x)), 0.001),
                       ("alpha", lambda x: fv.historical_var(EUR_BOOK, MKT, RETS, x), 0.01),
                       ("horizon", lambda x: fv.historical_var(EUR_BOOK, MKT, RETS, 0.01, x), 1.0),
                       ("decay", lambda x: fv.historical_var(EUR_BOOK, MKT, RETS, 0.01,
                                                             method="age", decay=x), 0.99),
                       ("lambda", lambda x: fv.historical_var(EUR_BOOK, MKT, RETS, 0.01,
                                                              method="fhs", ewma_lambda=x), 0.94)],
    "ewma_volatility": [("returns", lambda x: fv.ewma_volatility(_rets_with(x)), 0.001),
                        ("lambda", lambda x: fv.ewma_volatility(RETS, x), 0.94)],
    "parametric_var": [("returns", lambda x: fv.parametric_var(EUR_BOOK, MKT, _rets_with(x)), 0.001),
                       ("df", lambda x: fv.parametric_var(EUR_BOOK, MKT, RETS, dist="t", df=x), 5.0)],
    "var_covar": [("exposure", lambda x: fv.var_covar(pd.Series({"a": x, "b": 1.0}), COV2), 1e6),
                  ("cov", lambda x: fv.var_covar(W2, _cov_with(x)), 1e-5),
                  ("mean", lambda x: fv.var_covar(W2, COV2, mean=x), 0.0)],
    "sample_cov": [("returns", lambda x: fv.sample_cov(_rets_with(x)), 0.001)],
    "ewma_cov": [("returns", lambda x: fv.ewma_cov(_rets_with(x)), 0.001),
                 ("lambda", lambda x: fv.ewma_cov(RETS, x), 0.94)],
    "cornish_fisher_z": [("z", lambda x: fv.cornish_fisher_z(x, 0.1, 0.5), 2.0),
                         ("skew", lambda x: fv.cornish_fisher_z(2.0, x, 0.5), 0.1)],
    "cornish_fisher_valid": [("skew", lambda x: fv.cornish_fisher_valid(x, 0.0, 0.01), 0.1),
                             ("alpha", lambda x: fv.cornish_fisher_valid(0.1, 0.0, x), 0.01)],
    "cornish_fisher_var": [("sigma", lambda x: fv.cornish_fisher_var(x, 0.0, 0.0, 0.01), 1.0),
                           ("kurt", lambda x: fv.cornish_fisher_var(1.0, 0.0, x, 0.01), 0.5)],
    "monte_carlo_var": [("cov", lambda x: fv.monte_carlo_var(
        EUR_BOOK, MKT, pd.DataFrame([[x]], index=["FX:EUR"], columns=["FX:EUR"]),
        n_scenarios=200, n_boot=5), 3.6e-5),
                        ("df", lambda x: fv.monte_carlo_var(EUR_BOOK, MKT, COV1, dist="t", df=x,
                                                            n_scenarios=200, n_boot=5), 5.0)],
    "simulate_factor_returns": [("cov", lambda x: fv.simulate_factor_returns(_cov_with(x), 10), 1e-5),
                                ("horizon", lambda x: fv.simulate_factor_returns(
                                    COV2, 10, horizon_days=x), 1.0)],
    "JumpSpec": [("prob", lambda x: fv.JumpSpec(prob=x, mean={"FX:EUR": -0.1}), 0.01),
                 ("mean", lambda x: fv.JumpSpec(prob=0.01, mean={"FX:EUR": x}), -0.1)],
    "robust_cholesky": [("cov", lambda x: fv.robust_cholesky(_cov_with(x).to_numpy()), 1e-5)],
    "nearest_correlation_higham": [("corr", lambda x: fv.nearest_correlation_higham(
        np.array([[1.0, x], [x, 1.0]])), 0.5)],
    "var_standard_error": [("pnl", lambda x: fv.var_standard_error(_arr_with(PNL, x), 0.01), 0.0)],
    "var_standard_error_bootstrap": [("pnl", lambda x: fv.var_standard_error_bootstrap(
        _arr_with(PNL, x), 0.01, n_boot=5), 0.0)],
    "empirical_var": [("pnl", lambda x: fv.empirical_var(_arr_with(PNL, x)), 0.0),
                      ("weights", lambda x: fv.empirical_var(PNL, 0.01, _arr_with(np.ones(300), x)), 1.0)],
    "empirical_es": [("pnl", lambda x: fv.empirical_es(_arr_with(PNL, x)), 0.0)],
    "empirical_var_es": [("pnl", lambda x: fv.empirical_var_es(_arr_with(PNL, x)), 0.0),
                         ("alpha", lambda x: fv.empirical_var_es(PNL, x), 0.01)],
    "normal_var": [("sigma", lambda x: fv.normal_var(x, 0.01), 1.0)],
    "normal_es": [("mean", lambda x: fv.normal_es(1.0, 0.01, x), 0.0)],
    "t_var": [("df", lambda x: fv.t_var(1.0, 0.01, x), 5.0)],
    "t_es": [("sigma", lambda x: fv.t_es(x, 0.01, 5.0), 1.0)],
    "kupiec_pof": [("n_obs", lambda x: fv.kupiec_pof(n_obs=x, n_exceptions=2, alpha=0.01), 250),
                   ("alpha", lambda x: fv.kupiec_pof(n_obs=250, n_exceptions=2, alpha=x), 0.01)],
    "christoffersen_independence": [("e", lambda x: fv.christoffersen_independence([0, 1, x, 0]), 0)],
    "conditional_coverage": [("e", lambda x: fv.conditional_coverage([0, 1, x, 0], 0.01), 0)],
    "basel_traffic_light": [("n_exceptions", lambda x: fv.basel_traffic_light(
        n_obs=250, n_exceptions=x), 4)],
    "evaluate_var_backtest": [("pnl", lambda x: fv.evaluate_var_backtest(
        _arr_with(PNL, x), np.full(300, 2e5), 0.01), 0.0)],
    "rolling_backtest": [("returns", _rolling, 0.001)],
    "es_backtest_acerbi_szekely": [("pnl", lambda x: fv.es_backtest_acerbi_szekely(
        _arr_with(PNL, x), np.full(300, 2e5), np.full(300, 2.5e5), n_sim=5), 0.0)],
    "acerbi_szekely_z2": [("pnl", lambda x: fv.acerbi_szekely_z2(
        _arr_with(PNL, x), np.full(300, 2e5), np.full(300, 2.5e5), alpha=0.025, n_sims=5), 0.0),
                          ("es", lambda x: fv.acerbi_szekely_z2(
        PNL, np.full(300, 2e5), _arr_with(np.full(300, 2.5e5), x), alpha=0.025, n_sims=5), 2.5e5)],
    "Scenario": [("shock", lambda x: fv.Scenario("s", {"FX:EUR": x}), 0.01)],
    "usd_broad_move": [("pct", lambda x: fv.usd_broad_move(["EUR"], x), 0.1)],
    "peg_break_scenario": [("jump", lambda x: fv.peg_break_scenario("HKD", x), -0.3)],
    "run_stress": [("shock", lambda x: fv.run_stress(
        EUR_BOOK, MKT, {"s": fv.Scenario("s", {"FX:EUR": 0.01})}, option_method=x), "full")],
    "sensitivity_ladder": [("shock", lambda x: fv.sensitivity_ladder(
        EUR_BOOK, MKT, "FX:EUR", [0.0, x]), 0.01)],
    "reverse_stress_linear": [("radius", lambda x: fv.reverse_stress_linear(W2, COV2, radius=x), 2.0),
                              ("exposure", lambda x: fv.reverse_stress_linear(
                                  pd.Series({"a": x, "b": 1.0}), COV2, radius=2.0), 1e6)],
    "reverse_stress_numerical": [("radius", lambda x: fv.reverse_stress_numerical(W2, COV2, x), 2.0)],
}

#: public names that are not input-taking entry points, with the reason
EXEMPT = {
    "TRADING_DAYS_PER_YEAR": "constant",
    "PEG_VOL_THRESHOLD": "constant",
    "PegBlindnessWarning": "warning class",
    "NumericalWarning": "warning class",
    "Position": "type alias",
    "HistoricalVaRResult": "output record",
    "ParametricVaRResult": "output record",
    "MonteCarloVaRResult": "output record",
    "TrafficLight": "output record",
    "BacktestResult": "output record",
    "historical_scenarios": "takes no input",
    "cornish_fisher_domain_ok": "deprecated; keeps its old False-for-NaN result for one release",
}

#: run_stress takes no numeric input of its own; it is fuzzed through a
#: non-finite scenario shock (Scenario raises first) and a bad method name.
STRING_SITES = {("run_stress", "shock"), ("split_pair", "pair"), ("fx_factor", "ccy"),
                ("ir_factor", "ccy"), ("vol_factor", "pair")}


def test_registry_covers_public_api():
    missing = [n for n in fv.__all__ if n not in CASES and n not in EXEMPT]
    assert not missing, f"public callables without a fuzz entry: {missing}"
    stale = [n for n in list(CASES) + list(EXEMPT) if n not in fv.__all__]
    assert not stale, f"registry names not in fx_var.__all__: {stale}"


SITES = [(name, label, f, ok) for name, cases in CASES.items() for label, f, ok in cases]


@pytest.mark.parametrize("name, label, f, ok", SITES, ids=[f"{n}-{lab}" for n, lab, _, _ in SITES])
def test_valid_call_succeeds(name, label, f, ok):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f(ok)


def _inject(site, bad):
    name, label, f, _ = site
    if (name, label) in STRING_SITES:
        # string arguments: NaN float, empty string, malformed code
        bad = float("nan") if np.isnan(bad) else ("" if bad > 0 else "XX")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            f(bad)


@pytest.mark.parametrize("bad", BAD, ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("site", SITES, ids=[f"{n}-{lab}" for n, lab, _, _ in SITES])
def test_every_site_rejects_every_non_finite_value(site, bad):
    _inject(site, bad)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(site=st.sampled_from(SITES), bad=st.sampled_from(BAD))
def test_non_finite_input_raises_value_error(site, bad):
    _inject(site, bad)


@settings(max_examples=200, deadline=None)
@given(pnl=st.lists(st.floats(-1e9, 1e9), min_size=1, max_size=300),
       alpha=st.floats(1e-6, 0.4999))
def test_empirical_var_es_finite_and_ordered(pnl, alpha):
    var, es = fv.empirical_var_es(np.array(pnl), alpha)
    assert np.isfinite(var) and np.isfinite(es)
    assert es >= var - 1e-9 * max(1.0, abs(var))


@settings(max_examples=100, deadline=None)
@given(alpha=st.one_of(st.floats(0.5, 1e6), st.floats(-1e6, 0.0)))
def test_alpha_outside_open_half_interval_raises(alpha):
    with pytest.raises(ValueError, match="tail probability"):
        fv.empirical_var(PNL, alpha)
