"""Differential harness: every compiled engine against the Python reference.

For each of the four engines with compiled ports (equity options, FX
options, equity VaR, FX VaR) this draws 10,000 random inputs and checks
that ``engine="cpp"`` and ``engine="rust"`` agree with ``engine="python"``
to the cross-language tolerances of ``bindings/*/tests/test_binding.py``.
The inputs cover the edge regions of the Round 1 differential tests: K/S in
[0.3, 3], expiries from 1e-3 to 10 years, vols from 0.01 to 1.5, negative
rates, tail probabilities from 0.001 to 0.25 and sample sizes from 50 to 400.

A compiled engine that is not installed is skipped, so this file also runs
on a pure-Python checkout; CI installs both extensions, so there it runs in
full. The switch's own error contract (unknown name, missing extension) is
tested without any extension.
"""

from __future__ import annotations

import importlib
import math
import sys

import numpy as np
import pytest

engines = pytest.importorskip("qp_core.engines")
eq_engine = pytest.importorskip("eq_options.engine")
fx_engine = pytest.importorskip("fx_options.engine")
eqv_engine = pytest.importorskip("eq_var.engine")
fxv_engine = pytest.importorskip("fx_var.engine")

N = 10_000
EPS = np.finfo(float).eps
COMPILED = [pytest.param(e, marks=pytest.mark.skipif(not engines.engine_available(e),
                                                     reason=f"{e} extension not installed"))
            for e in ("cpp", "rust")]


# ------------------------------------------------------------- switch contract


@pytest.mark.parametrize("mod,call", [
    (eq_engine, lambda m, e: m.bs_price(100, 100, 1, 0.01, 0.2, engine=e)),
    (fx_engine, lambda m, e: m.gk_price(1.1, 1.1, 1, 0.02, 0.01, 0.1, "call", engine=e)),
    (eqv_engine, lambda m, e: m.historical_var(np.arange(100.0), engine=e)),
    (fxv_engine, lambda m, e: m.normal_var(1e4, engine=e)),
])
def test_unknown_engine_raises_value_error(mod, call):
    with pytest.raises(ValueError, match="unknown engine 'fortran'"):
        call(mod, "fortran")


@pytest.mark.parametrize("engine,module", [("cpp", "qp_cpp"), ("rust", "qp_rust")])
def test_missing_extension_raises_import_error_with_hint(engine, module, monkeypatch):
    monkeypatch.setitem(sys.modules, module, None)  # makes the import fail
    with pytest.raises(ImportError, match=f"pip install ./bindings/"):
        eq_engine.bs_price(100, 100, 1, 0.01, 0.2, engine=engine)
    assert engines.engine_available(engine) is False


def test_python_engine_is_the_reference_unchanged():
    eq = importlib.import_module("eq_options")
    args = (101.0, 97.0, 0.7, 0.03, 0.22, 0.01, "put")
    assert eq_engine.bs_price(*args) == eq.bs_price(*args)
    assert eq_engine.bs_greeks(*args) == eq.bs_greeks(*args)


# ------------------------------------------------------------------- inputs


def _contracts(seed: int, n: int = N, fx: bool = False):
    """Columns S, K, T, r, q, sigma, is_call drawn as in test_binding.py."""
    rng = np.random.default_rng(seed)
    S = (1.25 if fx else 100.0) * np.exp(rng.uniform(-0.3, 0.3, n))
    K = S * np.exp(rng.uniform(-1.2, 1.1, n))
    T = np.exp(rng.uniform(math.log(1e-3), math.log(10.0), n))
    r = rng.uniform(-0.02, 0.08, n)
    q = rng.uniform(-0.02, 0.06, n)
    sigma = np.exp(rng.uniform(math.log(0.01), math.log(1.5), n))
    is_call = rng.random(n) < 0.5
    return S, K, T, r, q, sigma, is_call


def _closed_form_tol(S, K, T, r, q, sigma, ref):
    """64 eps (1 + d1^2) (|ref| + max(S, K)): see test_binding.py."""
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q) * T) / v + 0.5 * v
    return 64.0 * EPS * (1.0 + d1 * d1) * (abs(ref) + max(S, K))


def _iv_tol(S, K, T, q, sigma, d1):
    """4 max(eps max(S, K) / vega, 1e-12 sigma): see test_binding.py."""
    if sigma <= 0.0:
        return 1e-12
    vega = S * math.exp(-q * T - 0.5 * d1 * d1) * math.sqrt(T / (2 * math.pi))
    return 4.0 * max(EPS * max(S, K) / vega if vega > 0 else math.inf, 1e-12 * sigma)


def _sides(is_call):
    return ["call" if c else "put" for c in is_call]


# ----------------------------------------------------------- equity options


@pytest.mark.parametrize("engine", COMPILED)
def test_eq_options_10k(engine):
    S, K, T, r, q, sigma, is_call = _contracts(101)
    got = eq_engine.bs_price_batch(S, K, T, r, sigma, q, is_call, engine=engine)
    n_iv = 0
    for i, side in enumerate(_sides(is_call)):
        a = (S[i], K[i], T[i], r[i], sigma[i], q[i], side)
        ref = eq_engine.bs_price(*a)
        assert abs(got[i] - ref) <= _closed_form_tol(S[i], K[i], T[i], r[i], q[i], sigma[i], ref), a
        assert eq_engine.bs_price(*a, engine=engine) == got[i]  # batch == scalar
        g_ref = eq_engine.bs_greeks(*a)
        g = eq_engine.bs_greeks(*a, engine=engine)
        for name, want in vars(g_ref).items():
            tol = _closed_form_tol(S[i], K[i], T[i], r[i], q[i], sigma[i], want) * max(1.0, 1.0 / T[i])
            assert abs(getattr(g, name) - want) <= tol, (name, a)
        try:
            iv_ref = eq_engine.implied_vol(ref, S[i], K[i], T[i], r[i], q[i], side)
        except ValueError:
            with pytest.raises(ValueError):
                eq_engine.implied_vol(ref, S[i], K[i], T[i], r[i], q[i], side, engine=engine)
            continue
        if ref < EPS * max(S[i], K[i]):
            # Quote below the price resolution eps max(S, K): the vol is not
            # determined to any tolerance, so a compiled refusal is accepted.
            try:
                eq_engine.implied_vol(ref, S[i], K[i], T[i], r[i], q[i], side, engine=engine)
            except ValueError:
                pass
            continue
        iv = eq_engine.implied_vol(ref, S[i], K[i], T[i], r[i], q[i], side, engine=engine)
        if iv_ref > 0:
            v = iv_ref * math.sqrt(T[i])
            d1 = (math.log(S[i] / K[i]) + (r[i] - q[i]) * T[i]) / v + 0.5 * v
        else:
            d1 = 0.0
        assert abs(iv - iv_ref) <= _iv_tol(S[i], K[i], T[i], q[i], iv_ref, d1), a
        n_iv += 1
    # About 43 % of the grid is invertible by the reference (deep OTM and short
    # expiries are refused or below price resolution); require 40 %.
    assert n_iv > 0.4 * N


@pytest.mark.parametrize("engine", COMPILED)
def test_eq_crr_sample(engine):
    # Trees cost O(n^2) in Python, so 200 of the 10,000 contracts, 200 steps.
    S, K, T, r, q, sigma, is_call = _contracts(102, 200)
    for i, side in enumerate(_sides(is_call)):
        for style in ("european", "american"):
            a = (S[i], K[i], T[i], r[i], sigma[i], q[i], side, style, 200)
            assert abs(eq_engine.crr_price(*a, engine=engine) - eq_engine.crr_price(*a)) \
                <= 4.0 * 200 * EPS * max(S[i], K[i]), a


# --------------------------------------------------------------- FX options


@pytest.mark.parametrize("engine", COMPILED)
def test_fx_options_10k(engine):
    S, K, T, rd, rf, sigma, is_call = _contracts(201, fx=True)
    got = fx_engine.gk_price_batch(S, K, T, rd, rf, sigma, is_call, engine=engine)
    n_iv = 0
    for i, side in enumerate(_sides(is_call)):
        a = (S[i], K[i], T[i], rd[i], rf[i], sigma[i], side)
        ref = fx_engine.gk_price(*a)
        assert abs(got[i] - ref) <= _closed_form_tol(S[i], K[i], T[i], rd[i], rf[i], sigma[i], ref), a
        g_ref = fx_engine.analytic_greeks(*a)
        g = fx_engine.analytic_greeks(*a, engine=engine)
        for name, want in vars(g_ref).items():
            tol = _closed_form_tol(S[i], K[i], T[i], rd[i], rf[i], sigma[i], want) * max(1.0, 1.0 / T[i])
            assert abs(getattr(g, name) - want) <= tol, (name, a)
        try:
            iv_ref = fx_engine.implied_vol(ref, S[i], K[i], T[i], rd[i], rf[i], side)
        except ValueError:
            with pytest.raises(ValueError):
                fx_engine.implied_vol(ref, S[i], K[i], T[i], rd[i], rf[i], side, engine=engine)
            continue
        if ref < EPS * max(S[i], K[i]):  # below price resolution: see equity test
            try:
                fx_engine.implied_vol(ref, S[i], K[i], T[i], rd[i], rf[i], side, engine=engine)
            except ValueError:
                pass
            continue
        iv = fx_engine.implied_vol(ref, S[i], K[i], T[i], rd[i], rf[i], side, engine=engine)
        if iv_ref > 0:
            v = iv_ref * math.sqrt(T[i])
            d1 = (math.log(S[i] / K[i]) + (rd[i] - rf[i]) * T[i]) / v + 0.5 * v
        else:
            d1 = 0.0
        assert abs(iv - iv_ref) <= _iv_tol(S[i], K[i], T[i], rf[i], iv_ref, d1), a
        n_iv += 1
    # About 43 % of the grid is invertible by the reference (deep OTM and short
    # expiries are refused or below price resolution); require 40 %.
    assert n_iv > 0.4 * N


@pytest.mark.parametrize("engine", COMPILED)
def test_fx_binomial_sample(engine):
    S, K, T, rd, rf, sigma, is_call = _contracts(202, 200, fx=True)
    for i, side in enumerate(_sides(is_call)):
        for style in ("european", "american"):
            a = (S[i], K[i], T[i], rd[i], rf[i], sigma[i], side, 200, style)
            assert abs(fx_engine.binomial_price(*a, engine=engine) - fx_engine.binomial_price(*a)) \
                <= 4.0 * 200 * EPS * max(S[i], K[i]), a


# ------------------------------------------------------------------- VaR


def _var_cases(seed: int, n: int = N):
    """(pnl, alpha) pairs: Student-t(4) P&L, n in [50, 400], alpha in [0.001, 0.25]."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        size = int(rng.integers(50, 401))
        yield rng.standard_t(4, size=size) * 1e4, float(np.exp(rng.uniform(math.log(1e-3),
                                                                          math.log(0.25))))


def _cov_case(rng):
    d = int(rng.integers(1, 7))
    a = rng.normal(size=(d, d)) * 0.01
    return rng.normal(size=d) * 1e6, a @ a.T + np.eye(d) * 1e-5


@pytest.mark.parametrize("engine", COMPILED)
def test_eq_var_10k(engine):
    for pnl, alpha in _var_cases(301):
        assert eqv_engine.historical_var(pnl, alpha, engine=engine) == \
            eqv_engine.historical_var(pnl, alpha)
        assert eqv_engine.expected_shortfall(pnl, alpha, engine=engine) == \
            pytest.approx(eqv_engine.expected_shortfall(pnl, alpha), rel=1e-13)
    rng = np.random.default_rng(302)
    for _ in range(N):
        w, cov = _cov_case(rng)
        alpha = float(rng.uniform(0.001, 0.25))
        dist = "t" if rng.random() < 0.5 else "normal"
        df, mean, h = float(rng.uniform(2.5, 30.0)), float(rng.normal() * 100), int(rng.integers(1, 11))
        # VaR = z sd - mean can cancel (seed 302 has 48 - 46 = 1.76), so the
        # 1e-12 relative bound applies to the terms: |z| <= 10 here.
        sd = math.sqrt(float(w @ cov @ w) * h)
        assert eqv_engine.parametric_var(w, cov, alpha, dist, df, mean, h, engine=engine) == \
            pytest.approx(eqv_engine.parametric_var(w, cov, alpha, dist, df, mean, h), rel=1e-12,
                          abs=1e-12 * (10.0 * sd + abs(mean)))


@pytest.mark.parametrize("engine", COMPILED)
def test_fx_var_10k(engine):
    for pnl, alpha in _var_cases(401):
        var, es = fxv_engine.empirical_var_es(pnl, alpha, engine=engine)
        var_ref, es_ref = fxv_engine.empirical_var_es(pnl, alpha)
        assert var == var_ref
        assert es == pytest.approx(es_ref, rel=1e-13)
    rng = np.random.default_rng(402)
    for _ in range(N):
        sigma = float(np.exp(rng.uniform(0.0, math.log(1e7))))
        alpha = float(rng.uniform(0.001, 0.25))
        mean, df = float(rng.normal() * 1e3), float(rng.uniform(2.5, 30.0))
        for fn, extra in ((fxv_engine.normal_var, ()), (fxv_engine.normal_es, ()),
                          (fxv_engine.t_var, (df,)), (fxv_engine.t_es, (df,))):
            assert fn(sigma, alpha, *extra, mean, engine=engine) == \
                pytest.approx(fn(sigma, alpha, *extra, mean), rel=1e-12, abs=1e-9)
    for _ in range(N // 10):  # var_covar goes through pandas in Python: 1,000 cases
        w, cov = _cov_case(rng)
        alpha, h = float(rng.uniform(0.001, 0.25)), float(rng.uniform(1.0, 10.0))
        dist = "t" if rng.random() < 0.5 else "normal"
        args = (w, cov, alpha, h, dist, float(rng.uniform(2.5, 30.0)), 0.0)
        assert fxv_engine.var_covar(*args, engine=engine) == \
            pytest.approx(fxv_engine.var_covar(*args), rel=1e-12)
