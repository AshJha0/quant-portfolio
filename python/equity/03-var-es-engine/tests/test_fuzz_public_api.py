"""Spec S12: every public entry point refuses NaN, +inf, -inf, empty input and
out-of-domain values with ``ValueError``. It never returns a number.

``REGISTRY`` has one entry per name in ``eq_var.__all__``. A new public name
without an entry fails ``test_registry_covers_public_api``, so the fuzz test
cannot fall behind the API. Each callable entry gives a valid call and the
arguments to poison. Hypothesis picks the argument, the bad value and, for
arrays, the position of the bad entry.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import eq_var as ev
from eq_var.data import demo_covariance, demo_portfolio

NON_FINITE = [np.nan, np.inf, -np.inf]
BAD_ALPHA = [0.0, 0.5, 0.99, -0.01, 1.0]

RNG = np.random.default_rng(1234)
PNL = RNG.normal(0.0, 100.0, 300)
PF = demo_portfolio()
COV = demo_covariance()
W = PF.delta_exposures()
RETS = RNG.multivariate_normal(np.zeros(4), COV, size=300)
EXC = (np.arange(300) % 37 == 0)


@dataclass
class Entry:
    call: Callable[..., Any] | None
    valid: dict[str, Any] = field(default_factory=dict)
    poison: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()  # args where a negative value is invalid
    no_empty: tuple[str, ...] = ()  # array args where an empty array is valid
    note: str = ""


def _factor(level):
    return ev.RiskFactor("A", "equity", level)


def _option(**kw):
    base = dict(name="o", underlier="SPX", vol_factor="SPX_IV", strike=4750.0, expiry=0.25, rate=0.03,
                div_yield=0.015, kind="put", contracts=2.0, multiplier=100.0)
    base.update(kw)
    return ev.Portfolio([ev.OptionPosition(**base)], dict(PF.factors))


def _scenario(shock):
    return ev.StressScenario("x", "fuzz", {"index": shock})


REGISTRY: dict[str, Entry] = {
    # portfolio
    "RiskFactor": Entry(lambda level: _factor(level), {"level": 100.0}, ("level",), ("level",)),
    "Position": Entry(None, note="abstract base class; subclasses are fuzzed"),
    "EquityPosition": Entry(
        lambda shares: ev.Portfolio([ev.EquityPosition(name="a", factor="A", shares=shares)], {"A": _factor(1.0)}),
        {"shares": 10.0}, ("shares",)),
    "FuturePosition": Entry(
        lambda contracts, multiplier: ev.Portfolio(
            [ev.FuturePosition(name="f", factor="A", contracts=contracts, multiplier=multiplier)], {"A": _factor(1.0)}),
        {"contracts": 1.0, "multiplier": 50.0}, ("contracts", "multiplier")),
    "OptionPosition": Entry(lambda **kw: _option(**kw),
                            {"strike": 4750.0, "expiry": 0.25, "rate": 0.03, "div_yield": 0.015, "contracts": 2.0},
                            ("strike", "expiry", "rate", "div_yield", "contracts"), ("strike", "expiry")),
    "Portfolio": Entry(lambda scenarios: PF.pnl(scenarios), {"scenarios": RETS[:20]}, ("scenarios",)),
    "bs_price": Entry(ev.bs_price, dict(spot=100.0, strike=100.0, tau=0.5, rate=0.01, div_yield=0.0, vol=0.2,
                                        kind="call"),
                      ("spot", "strike", "tau", "rate", "div_yield", "vol"), ("spot", "strike", "tau", "vol")),
    "bs_greeks": Entry(ev.bs_greeks, dict(spot=100.0, strike=100.0, tau=0.5, rate=0.01, div_yield=0.0, vol=0.2,
                                          kind="call"),
                       ("spot", "strike", "tau", "rate", "div_yield", "vol"), ("spot", "strike", "tau", "vol")),
    # historical
    "historical_var": Entry(ev.historical_var, {"pnl": PNL, "alpha": 0.01}, ("pnl", "alpha")),
    "age_weighted_var": Entry(ev.age_weighted_var, {"pnl": PNL, "alpha": 0.01, "lam": 0.98}, ("pnl", "alpha", "lam")),
    "age_weighted_es": Entry(ev.age_weighted_es, {"pnl": PNL, "alpha": 0.01, "lam": 0.98}, ("pnl", "alpha", "lam")),
    "brw_weights": Entry(ev.brw_weights, {"n": 50, "lam": 0.98}, ("n", "lam"), ("n",)),
    "ewma_volatility": Entry(ev.ewma_volatility, {"x": PNL, "lam": 0.94}, ("x", "lam")),
    "ewma_forecast": Entry(ev.ewma_forecast, {"x": PNL, "lam": 0.94}, ("x", "lam")),
    "filtered_historical_scenarios": Entry(ev.filtered_historical_scenarios, {"factor_returns": RETS, "lam": 0.94},
                                           ("factor_returns", "lam")),
    "factor_filtered_historical_var": Entry(
        lambda factor_returns, alpha, lam: ev.factor_filtered_historical_var(PF, factor_returns, alpha, lam),
        {"factor_returns": RETS, "alpha": 0.01, "lam": 0.94}, ("factor_returns", "alpha", "lam")),
    "filtered_historical_var": Entry(ev.filtered_historical_var, {"pnl": PNL, "alpha": 0.01, "lam": 0.94},
                                     ("pnl", "alpha", "lam")),
    "scale_var_sqrt_time": Entry(ev.scale_var_sqrt_time, {"var_1d": 5.0, "horizon_days": 10},
                                 ("var_1d", "horizon_days"), ("var_1d", "horizon_days")),
    "overlapping_horizon_pnl": Entry(ev.overlapping_horizon_pnl, {"pnl": PNL, "horizon_days": 10},
                                     ("pnl", "horizon_days"), ("horizon_days",)),
    # parametric
    "sample_covariance": Entry(ev.sample_covariance, {"returns": RETS}, ("returns",)),
    "ewma_covariance": Entry(ev.ewma_covariance, {"returns": RETS, "lam": 0.94}, ("returns", "lam")),
    "portfolio_sigma": Entry(ev.portfolio_sigma, {"exposures": W, "cov": COV}, ("exposures", "cov")),
    "parametric_var": Entry(ev.parametric_var, {"exposures": W, "cov": COV, "alpha": 0.01, "dist": "t", "df": 6.0,
                                                "mean": 0.0, "horizon_days": 1},
                            ("exposures", "cov", "alpha", "df", "mean", "horizon_days"), ("horizon_days",)),
    "cornish_fisher_z": Entry(ev.cornish_fisher_z, {"z": np.array([-2.0, 0.0, 1.0]), "skew": -0.3, "excess_kurt": 1.0},
                              ("z", "skew", "excess_kurt")),
    "cornish_fisher_valid": Entry(ev.cornish_fisher_valid, {"skew": -0.3, "excess_kurt": 1.0, "alpha": 0.01},
                                  ("skew", "excess_kurt", "alpha")),
    "cornish_fisher_domain_ok": Entry(ev.cornish_fisher_domain_ok, {"skew": -0.3, "excess_kurt": 1.0, "z_range": 3.5},
                                      ("skew", "excess_kurt", "z_range"), ("z_range",)),
    "cornish_fisher_var": Entry(ev.cornish_fisher_var, {"sigma": 1.0, "alpha": 0.01, "skew": -0.3, "excess_kurt": 1.0,
                                                        "mean": 0.0},
                                ("sigma", "alpha", "skew", "excess_kurt", "mean"), ("sigma",)),
    # monte carlo
    "cholesky": Entry(ev.cholesky, {"cov": COV}, ("cov",)),
    "higham_nearest_correlation": Entry(ev.higham_nearest_correlation, {"corr": np.eye(3)}, ("corr",)),
    "repair_covariance": Entry(ev.repair_covariance, {"cov": COV, "method": "clip"}, ("cov",)),
    "safe_cholesky": Entry(ev.safe_cholesky, {"cov": COV}, ("cov",)),
    "simulate_factor_returns": Entry(ev.simulate_factor_returns, {"cov": COV, "n_paths": 50, "dist": "t", "df": 5.0},
                                     ("cov", "n_paths", "df"), ("n_paths",)),
    "monte_carlo_pnl": Entry(lambda cov, n_paths, df: ev.monte_carlo_pnl(PF, cov, n_paths, "t", df),
                             {"cov": COV, "n_paths": 50, "df": 5.0}, ("cov", "n_paths", "df"), ("n_paths",)),
    "monte_carlo_var": Entry(lambda cov, alpha, df: ev.monte_carlo_var(PF, cov, alpha, 200, "t", df),
                             {"cov": COV, "alpha": 0.01, "df": 5.0}, ("cov", "alpha", "df")),
    "monte_carlo_risk": Entry(lambda cov, alpha, n_boot: ev.monte_carlo_risk(PF, cov, alpha, 200, n_boot=n_boot),
                              {"cov": COV, "alpha": 0.01, "n_boot": 5}, ("cov", "alpha", "n_boot"), ("n_boot",)),
    "var_standard_error_bootstrap": Entry(ev.var_standard_error_bootstrap, {"pnl": PNL, "alpha": 0.05, "n_boot": 5},
                                          ("pnl", "alpha", "n_boot"), ("n_boot",)),
    "var_standard_error_kde": Entry(ev.var_standard_error_kde, {"pnl": PNL, "alpha": 0.05}, ("pnl", "alpha")),
    "var_confidence_interval": Entry(ev.var_confidence_interval, {"pnl": PNL, "alpha": 0.05, "conf": 0.9},
                                     ("pnl", "alpha", "conf")),
    # expected shortfall
    "expected_shortfall": Entry(ev.expected_shortfall, {"pnl": PNL, "alpha": 0.025}, ("pnl", "alpha")),
    "normal_es": Entry(ev.normal_es, {"sigma": 1.0, "alpha": 0.025, "mean": 0.0}, ("sigma", "alpha", "mean"),
                       ("sigma",)),
    "student_t_es": Entry(ev.student_t_es, {"sigma": 1.0, "alpha": 0.025, "df": 6.0, "mean": 0.0},
                          ("sigma", "alpha", "df", "mean"), ("sigma", "df")),
    "parametric_es": Entry(ev.parametric_es, {"exposures": W, "cov": COV, "alpha": 0.025, "dist": "t", "df": 6.0,
                                              "mean": 0.0, "horizon_days": 1},
                           ("exposures", "cov", "alpha", "df", "mean", "horizon_days"), ("horizon_days",)),
    "es_standard_error_bootstrap": Entry(ev.es_standard_error_bootstrap, {"pnl": PNL, "alpha": 0.05, "n_boot": 5},
                                         ("pnl", "alpha", "n_boot"), ("n_boot",)),
    # backtesting
    "exceptions_from_pnl": Entry(ev.exceptions_from_pnl, {"pnl": PNL, "var": 200.0}, ("pnl", "var"), ("var",)),
    "kupiec_pof": Entry(ev.kupiec_pof, {"n_obs": 250, "n_exceptions": 3, "alpha": 0.01},
                        ("n_obs", "n_exceptions", "alpha"), ("n_obs", "n_exceptions")),
    "christoffersen_independence": Entry(ev.christoffersen_independence, {"exceptions": EXC.astype(float)},
                                         ("exceptions",)),
    "christoffersen_cc": Entry(ev.christoffersen_cc, {"exceptions": EXC.astype(float), "alpha": 0.01},
                               ("exceptions", "alpha")),
    "basel_traffic_light": Entry(ev.basel_traffic_light, {"n_obs": 250, "n_exceptions": 3, "alpha": 0.01},
                                 ("n_obs", "n_exceptions", "alpha"), ("n_obs", "n_exceptions")),
    "basel_zone_probabilities": Entry(ev.basel_zone_probabilities, {"n_obs": 250, "alpha": 0.01}, ("n_obs", "alpha"),
                                      ("n_obs",)),
    "acerbi_szekely_z2": Entry(ev.acerbi_szekely_z2, {"pnl": PNL, "var": 200.0, "es": 250.0, "alpha": 0.025,
                                                      "n_sims": 5},
                               ("pnl", "var", "es", "alpha", "n_sims"), ("var", "es", "n_sims")),
    "exception_cluster_table": Entry(ev.exception_cluster_table, {"exceptions": EXC.astype(float)}, ("exceptions",),
                                     no_empty=("exceptions",)),
    "rolling_var_backtest": Entry(
        lambda pnl, window, alpha: ev.rolling_var_backtest(pnl, ev.historical_var, window, alpha),
        {"pnl": PNL, "window": 250, "alpha": 0.05}, ("pnl", "window", "alpha"), ("window",)),
    "BacktestResult": Entry(None, note="plain container built by rolling_var_backtest, which is fuzzed"),
    # stress
    "StressScenario": Entry(lambda shock: _scenario(shock), {"shock": -0.1}, ("shock",)),
    "HISTORICAL_SCENARIOS": Entry(None, note="constant dict of StressScenario"),
    "scenario_shock_vector": Entry(lambda shock: ev.scenario_shock_vector(PF, _scenario(shock)), {"shock": -0.1},
                                   ("shock",)),
    "apply_scenario": Entry(lambda shock: ev.apply_scenario(PF, _scenario(shock)), {"shock": -0.1}, ("shock",)),
    "scenario_table": Entry(lambda shock: ev.scenario_table(PF, {"x": _scenario(shock)}), {"shock": -0.1},
                            ("shock",)),
    "sensitivity_ladder": Entry(lambda shocks: ev.sensitivity_ladder(PF, "SPX", shocks),
                                {"shocks": np.array([-0.1, 0.0, 0.1])}, ("shocks",)),
    "reverse_stress_delta": Entry(ev.reverse_stress_delta, {"exposures": W, "cov": COV, "radius": 3.0},
                                  ("exposures", "cov", "radius"), ("radius",)),
    "reverse_stress_delta_gamma": Entry(
        ev.reverse_stress_delta_gamma,
        {"exposures": W, "gamma": PF.gamma_matrix(), "cov": COV, "radius": 3.0, "n_starts": 1},
        ("exposures", "gamma", "cov", "radius"), ("radius",)),
}

CALLABLE = sorted(k for k, e in REGISTRY.items() if e.call is not None)


def test_registry_covers_public_api():
    assert set(REGISTRY) == set(ev.__all__)
    for name, e in REGISTRY.items():
        assert e.call is not None or e.note, f"{name}: non-fuzzed entries need a reason"


def _invoke(entry: Entry, kwargs: dict[str, Any]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        return entry.call(**kwargs)


@pytest.mark.parametrize("name", CALLABLE)
def test_valid_call_succeeds(name):
    _invoke(REGISTRY[name], dict(REGISTRY[name].valid))


@pytest.mark.parametrize("name", CALLABLE)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_poisoned_input_raises(name, data):
    entry = REGISTRY[name]
    arg = data.draw(st.sampled_from(entry.poison), label="arg")
    kwargs = dict(entry.valid)
    good = kwargs[arg]
    choices = list(NON_FINITE)
    if arg == "alpha":
        choices += BAD_ALPHA
    if arg in entry.negative:
        choices.append(-5.0)
    if isinstance(good, np.ndarray) and arg not in entry.no_empty:
        choices.append("empty")
    bad = data.draw(st.sampled_from(choices), label="bad")
    if isinstance(good, np.ndarray):
        if bad == "empty":
            kwargs[arg] = np.array([]).reshape((0,) + good.shape[1:]) if good.ndim > 1 else np.array([])
        else:
            arr = good.astype(float).copy()
            i = data.draw(st.integers(0, arr.size - 1), label="index")
            arr.flat[i] = bad
            kwargs[arg] = arr
    else:
        kwargs[arg] = bad
    with pytest.raises(ValueError):
        _invoke(entry, kwargs)
