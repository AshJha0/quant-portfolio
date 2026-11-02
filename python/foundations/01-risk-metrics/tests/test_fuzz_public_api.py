"""Hypothesis fuzz over every public callable.

The registry below must name every callable in ``eq_risk_metrics.__all__``.
A new public function without an entry fails ``test_registry_covers_public_api``.

Properties checked for each entry:
1. Input containing NaN, +inf or -inf raises ValueError.
2. Finite input either raises ValueError or returns only finite numbers.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable

import eq_risk_metrics as rm
import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

finite = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False)
bad = st.sampled_from([np.nan, np.inf, -np.inf])
conf = st.floats(min_value=0.51, max_value=0.999)


def _series(xs: list[float]) -> pd.Series:
    return pd.Series(xs, index=pd.bdate_range("2020-01-01", periods=len(xs)), dtype=float)


# name -> (call(series, scalar_param), data strategy for valid input)
REGISTRY: dict[str, tuple[Callable[[pd.Series, float], object], st.SearchStrategy]] = {
    "simple_returns": (lambda s, c: rm.simple_returns(s), positive),
    "log_returns": (lambda s, c: rm.log_returns(s), positive),
    "annualised_volatility": (lambda s, c: rm.annualised_volatility(s), finite),
    "rolling_volatility": (lambda s, c: rm.rolling_volatility(s, window=3), finite),
    "ewma_volatility": (lambda s, c: rm.ewma_volatility(s), finite),
    "var_historical": (lambda s, c: rm.var_historical(s, c), finite),
    "var_parametric": (lambda s, c: rm.var_parametric(s, c), finite),
    "var_cornish_fisher": (lambda s, c: rm.var_cornish_fisher(s, c), finite),
    "expected_shortfall": (lambda s, c: rm.expected_shortfall(s, c), finite),
    "cornish_fisher_valid": (lambda s, c: rm.cornish_fisher_valid(float(np.sum(s.to_numpy())), float(s.to_numpy()[0]) if len(s) else 0.0, 1 - c), finite),
    "max_drawdown": (lambda s, c: rm.max_drawdown(s), positive),
    "sharpe_ratio": (lambda s, c: rm.sharpe_ratio(s), finite),
    "sortino_ratio": (lambda s, c: rm.sortino_ratio(s), finite),
    "normality_report": (lambda s, c: rm.normality_report(s), finite),
    "count_var_exceptions": (lambda s, c: rm.count_var_exceptions(s, 0.01), finite),
    "kupiec_pof_test": (lambda s, c: rm.kupiec_pof_test(s, 0.01, c), finite),
}


def _public_callables() -> set[str]:
    return {
        n for n in rm.__all__
        if callable(getattr(rm, n)) and not inspect.isclass(getattr(rm, n))
        and not isinstance(getattr(rm, n), type)
    } - {"KupiecResult", "DrawdownResult", "NormalityReport"}


def test_registry_covers_public_api() -> None:
    assert set(REGISTRY) == _public_callables()


def _numbers(out: object) -> list[float]:
    if isinstance(out, pd.Series):
        return [float(v) for v in out.to_numpy(dtype=float)]
    if isinstance(out, dict):
        vals: list[float] = []
        for v in out.values():
            if isinstance(v, (bool, np.bool_)):
                continue
            if isinstance(v, (int, float, np.floating, np.integer)):
                vals.append(float(v))
            elif isinstance(v, pd.Series) and v.dtype.kind == "f":
                vals.extend(v.to_numpy(dtype=float).tolist())
        return vals
    if isinstance(out, (bool, np.bool_)):
        return []
    return [float(out)]


SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.mark.parametrize("name", sorted(REGISTRY))
@SETTINGS
@given(data=st.data())
def test_non_finite_input_raises(name: str, data: st.DataObject) -> None:
    call, elem = REGISTRY[name]
    xs = data.draw(st.lists(elem, min_size=5, max_size=40))
    pos = data.draw(st.integers(0, len(xs) - 1))
    xs[pos] = data.draw(bad)
    c = data.draw(conf)
    with pytest.raises(ValueError):
        call(_series(xs), c)


@pytest.mark.parametrize("name", sorted(REGISTRY))
@SETTINGS
@given(data=st.data())
def test_finite_input_gives_finite_output_or_error(name: str, data: st.DataObject) -> None:
    call, elem = REGISTRY[name]
    xs = data.draw(st.lists(elem, min_size=0, max_size=40))
    c = data.draw(conf)
    try:
        out = call(_series(xs), c)
    except ValueError:
        return
    vals = _numbers(out)
    if name == "rolling_volatility":
        vals = vals[2:]  # the first window-1 entries are warm-up NaN by design
    assert all(np.isfinite(v) for v in vals), (name, xs, out)


@pytest.mark.parametrize("bad_conf", [np.nan, np.inf, 0.0, 1.0])
def test_bad_confidence_raises(bad_conf: float) -> None:
    s = _series([0.01, -0.02, 0.03, -0.01, 0.0])
    for fn in (rm.var_historical, rm.var_parametric, rm.var_cornish_fisher, rm.expected_shortfall):
        with pytest.raises(ValueError):
            fn(s, bad_conf)
    with pytest.raises(ValueError):
        rm.kupiec_pof_test(s, 0.01, bad_conf)
