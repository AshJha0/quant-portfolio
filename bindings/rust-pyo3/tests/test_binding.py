"""Binding vs Python reference engines.

Every exposed function is compared with the Python reference package
(eq_options, fx_options, eq_var, fx_var) over seeded random inputs and the
edge regions. Tolerances follow the cross-language differential tests
(python/*/01-options-pricing/tests/test_differential.py); each one is
stated where it is used.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

MODULE = "qp_rust"
qp = importlib.import_module(MODULE)

eq_options = pytest.importorskip("eq_options")
fx_options = pytest.importorskip("fx_options")
eq_var = pytest.importorskip("eq_var")
fx_var = pytest.importorskip("fx_var")

EPS = np.finfo(float).eps
N_RANDOM = 400


# ------------------------------------------------------------------ inputs


def _option_grid(seed: int, n: int = N_RANDOM, fx: bool = False) -> list[tuple]:
    """Random contracts plus the edge regions of the differential test:
    deep ITM/OTM (K/S in [0.3, 3]), short expiry (1e-3 y), long expiry (10 y),
    low and high vol (0.01 to 1.5), negative rates."""
    rng = np.random.default_rng(seed)
    base = 1.25 if fx else 100.0
    rows = []
    for _ in range(n):
        S = base * math.exp(rng.uniform(-0.3, 0.3))
        K = S * math.exp(rng.uniform(-1.2, 1.1))
        T = float(np.exp(rng.uniform(math.log(1e-3), math.log(10.0))))
        r = rng.uniform(-0.02, 0.08)
        q = rng.uniform(-0.02, 0.06)
        sigma = float(np.exp(rng.uniform(math.log(0.01), math.log(1.5))))
        side = "call" if rng.random() < 0.5 else "put"
        rows.append((S, K, T, r, q, sigma, side))
    return rows


def _d1(S, K, T, r, q, sigma):
    v = sigma * math.sqrt(T)
    return (math.log(S / K) + (r - q) * T) / v + 0.5 * v


def _closed_form_tol(S, K, T, r, q, sigma, ref):
    """Differential-test tolerance for closed-form prices and Greeks:
    64 eps (1 + d1^2) (|ref| + max(S, K)). The (1 + d1^2) factor is the
    condition number of erfc in the tails; 64 eps covers the few roundings
    in exp/log/sqrt/erfc. Observed Round 1 maxima: 1.1e-13 abs on unit-scale
    prices (S ~ 100), well inside this bound."""
    M = max(S, K)
    d1 = _d1(S, K, T, r, q, sigma)
    return 64.0 * EPS * (1.0 + d1 * d1) * (abs(ref) + M)


def _iv_tol(S, K, T, r, q, sigma):
    """Implied vol tolerance: price error eps*M mapped through 1/vega, floored
    at 1e-12 relative, times 4 (as in the differential test; Round 1 max
    2.1e-13). A reference vol of 0 (the reference returns 0 when the quote
    equals the zero-vol value) must be matched to 1e-12 absolute."""
    if sigma <= 0.0:
        return 1e-12
    d1 = _d1(S, K, T, r, q, sigma)
    vega = S * math.exp(-q * T - 0.5 * d1 * d1) * math.sqrt(T / (2 * math.pi))
    M = max(S, K)
    return 4.0 * max(EPS * M / vega if vega > 0 else math.inf, 1e-12 * sigma)


# ------------------------------------------------------------ module basics


def test_available():
    assert qp.available() is True


# ---------------------------------------------------------- equity options


@pytest.mark.parametrize("seed", [1, 2])
def test_eq_price_and_greeks_match_reference(seed):
    for S, K, T, r, q, sigma, side in _option_grid(seed):
        ref = eq_options.bs_price(S, K, T, r, sigma, q, side)
        got = qp.eq_bs_price(S, K, T, r, sigma, q, side)
        assert abs(got - ref) <= _closed_form_tol(S, K, T, r, q, sigma, ref)
        g_ref = eq_options.bs_greeks(S, K, T, r, sigma, q, side)
        g = qp.eq_bs_greeks(S, K, T, r, sigma, q, side)
        for name, val in g.items():
            want = getattr(g_ref, name)
            # Greeks share the closed-form bound; theta and volga carry an
            # extra factor of up to T^-1 and d1^2 in their magnitudes, which
            # |ref| in the bound already scales with.
            tol = _closed_form_tol(S, K, T, r, q, sigma, want) * max(1.0, 1.0 / T)
            assert abs(val - want) <= tol, (name, S, K, T, r, q, sigma, side, val, want)


def test_eq_implied_vol_round_trip_matches_reference():
    n_checked = 0
    for S, K, T, r, q, sigma, side in _option_grid(3):
        price = eq_options.bs_price(S, K, T, r, sigma, q, side)
        try:
            ref = eq_options.implied_vol(price, S, K, T, r, q, side)
        except ValueError:
            # Python refuses (price at a no-arbitrage bound): binding must too.
            with pytest.raises(ValueError):
                qp.eq_implied_vol(price, S, K, T, r, q, side)
            continue
        got = qp.eq_implied_vol(price, S, K, T, r, q, side)
        assert abs(got - ref) <= _iv_tol(S, K, T, r, q, ref)
        n_checked += 1
    assert n_checked > N_RANDOM // 2


@pytest.mark.parametrize("american", [False, True])
def test_eq_crr_matches_reference(american):
    for S, K, T, r, q, sigma, side in _option_grid(4, n=60):
        steps = 200
        ref = eq_options.crr_price(S, K, T, r, sigma, q, side,
                                   "american" if american else "european", steps)
        got = qp.eq_crr_price(S, K, T, r, sigma, q, side, american=american, n_steps=steps)
        # Backward induction: one rounding per node per step, so 4 n eps M
        # (differential test; Round 1 observed 0).
        assert abs(got - ref) <= 4.0 * steps * EPS * max(S, K)


def test_eq_mc_brackets_closed_form_and_is_reproducible():
    S, K, T, r, sigma, q = 100.0, 105.0, 0.75, 0.03, 0.25, 0.01
    bs = eq_options.bs_price(S, K, T, r, sigma, q, "put")
    for threads in (1, 4):
        a = qp.eq_mc_price(S, K, T, r, sigma, q, "put", n_paths=200_000, seed=7, threads=threads)
        b = qp.eq_mc_price(S, K, T, r, sigma, q, "put", n_paths=200_000, seed=7, threads=threads)
        assert a == b  # fixed for a given (seed, threads)
        # RNG streams differ from NumPy, so agreement is statistical: within
        # 5 standard errors of the closed form (fixed seed, so not flaky).
        assert abs(a["price"] - bs) <= 5.0 * a["std_error"]
        assert a["n_paths"] == 200_000


# --------------------------------------------------------------- FX options


@pytest.mark.parametrize("seed", [11, 12])
def test_fx_price_and_greeks_match_reference(seed):
    for S, K, T, r_d, r_f, sigma, side in _option_grid(seed, fx=True):
        ref = fx_options.gk_price(S, K, T, r_d, r_f, sigma, side)
        got = qp.fx_gk_price(S, K, T, r_d, r_f, sigma, side)
        assert abs(got - ref) <= _closed_form_tol(S, K, T, r_d, r_f, sigma, ref)
        g_ref = fx_options.analytic_greeks(S, K, T, r_d, r_f, sigma, side)
        for name, val in qp.fx_gk_greeks(S, K, T, r_d, r_f, sigma, side).items():
            want = getattr(g_ref, name)
            tol = _closed_form_tol(S, K, T, r_d, r_f, sigma, want) * max(1.0, 1.0 / T)
            assert abs(val - want) <= tol, (name, val, want)


def test_fx_implied_vol_matches_reference():
    n_checked = 0
    for S, K, T, r_d, r_f, sigma, side in _option_grid(13, fx=True):
        price = fx_options.gk_price(S, K, T, r_d, r_f, sigma, side)
        try:
            ref = fx_options.implied_vol(price, S, K, T, r_d, r_f, side)
        except ValueError:
            with pytest.raises(ValueError):
                qp.fx_implied_vol(price, S, K, T, r_d, r_f, side)
            continue
        got = qp.fx_implied_vol(price, S, K, T, r_d, r_f, side)
        assert abs(got - ref) <= _iv_tol(S, K, T, r_d, r_f, ref)
        n_checked += 1
    assert n_checked > N_RANDOM // 2


@pytest.mark.parametrize("american", [False, True])
def test_fx_binomial_matches_reference(american):
    for S, K, T, r_d, r_f, sigma, side in _option_grid(14, n=60, fx=True):
        steps = 200
        ref = fx_options.binomial_price(S, K, T, r_d, r_f, sigma, side, steps,
                                        "american" if american else "european")
        got = qp.fx_binomial_price(S, K, T, r_d, r_f, sigma, side, steps=steps,
                                   american=american)
        assert abs(got - ref) <= 4.0 * steps * EPS * max(S, K)


def test_fx_mc_brackets_closed_form_and_is_reproducible():
    args = (1.10, 1.12, 0.5, 0.04, 0.02, 0.09, "call")
    gk = fx_options.gk_price(*args)
    for threads in (1, 3):
        a = qp.fx_mc_price(*args, n_paths=200_000, seed=3, threads=threads)
        assert a == qp.fx_mc_price(*args, n_paths=200_000, seed=3, threads=threads)
        assert abs(a["price"] - gk) <= 5.0 * a["std_error"]


# --------------------------------------------------------------- batch API


def _eq_columns(seed, n=500):
    rows = _option_grid(seed, n)
    S, K, T, r, q, sigma, side = (np.array(c) for c in zip(*rows))
    return S, K, T, r, sigma, q, side == "call"


def test_eq_batch_equals_scalar_bit_for_bit():
    S, K, T, r, sigma, q, is_call = _eq_columns(21)
    prices = qp.eq_bs_price_batch(S, K, T, r, sigma, q, is_call)
    greeks = qp.eq_bs_greeks_batch(S, K, T, r, sigma, q, is_call)
    for i in range(S.size):
        side = "call" if is_call[i] else "put"
        assert prices[i] == qp.eq_bs_price(S[i], K[i], T[i], r[i], sigma[i], q[i], side)
        g = qp.eq_bs_greeks(S[i], K[i], T[i], r[i], sigma[i], q[i], side)
        for name, val in g.items():
            assert greeks[name][i] == val, name
    # Implied vol: keep the rows whose quote is strictly inside the
    # no-arbitrage bounds (the scalar call succeeds), then compare.
    keep = []
    for i in range(S.size):
        side = "call" if is_call[i] else "put"
        try:
            qp.eq_implied_vol(prices[i], S[i], K[i], T[i], r[i], q[i], side)
            keep.append(i)
        except ValueError:
            pass
    assert len(keep) > S.size // 2
    k = np.array(keep)
    iv = qp.eq_implied_vol_batch(prices[k], S[k], K[k], T[k], r[k], q[k], is_call[k])
    for j, i in enumerate(keep):
        side = "call" if is_call[i] else "put"
        assert iv[j] == qp.eq_implied_vol(prices[i], S[i], K[i], T[i], r[i], q[i], side)


def test_fx_batch_equals_scalar_bit_for_bit():
    rows = _option_grid(22, 500, fx=True)
    S, K, T, rd, rf, sigma, side = (np.array(c) for c in zip(*rows))
    is_call = side == "call"
    prices = qp.fx_gk_price_batch(S, K, T, rd, rf, sigma, is_call)
    greeks = qp.fx_gk_greeks_batch(S, K, T, rd, rf, sigma, is_call)
    for i in range(S.size):
        assert prices[i] == qp.fx_gk_price(S[i], K[i], T[i], rd[i], rf[i], sigma[i], side[i])
        g = qp.fx_gk_greeks(S[i], K[i], T[i], rd[i], rf[i], sigma[i], side[i])
        for name, val in g.items():
            assert greeks[name][i] == val, name
    keep = []
    for i in range(S.size):
        try:
            qp.fx_implied_vol(prices[i], S[i], K[i], T[i], rd[i], rf[i], side[i])
            keep.append(i)
        except ValueError:
            pass
    assert len(keep) > S.size // 2
    k = np.array(keep)
    iv = qp.fx_implied_vol_batch(prices[k], S[k], K[k], T[k], rd[k], rf[k], is_call[k])
    for j, i in enumerate(keep):
        assert iv[j] == qp.fx_implied_vol(prices[i], S[i], K[i], T[i], rd[i], rf[i], side[i])


def test_batch_does_not_modify_inputs_and_accepts_strided():
    S, K, T, r, sigma, q, is_call = _eq_columns(23, 64)
    keep = [a.copy() for a in (S, K, T, r, sigma, q)]
    out = qp.eq_bs_price_batch(S, K, T, r, sigma, q, is_call)
    for a, b in zip((S, K, T, r, sigma, q), keep):
        assert np.array_equal(a, b)
    # A strided view is converted, not rejected, and gives the same numbers.
    big = np.repeat(S, 2)
    strided = qp.eq_bs_price_batch(big[::2], K, T, r, sigma, q, is_call)
    assert np.array_equal(strided, out)


def test_batch_error_names_element():
    S, K, T, r, sigma, q, is_call = _eq_columns(24, 10)
    sigma[6] = np.nan
    with pytest.raises(ValueError, match="element 6"):
        qp.eq_bs_price_batch(S, K, T, r, sigma, q, is_call)
    with pytest.raises(ValueError, match="length mismatch"):
        qp.eq_bs_price_batch(S[:-1], K, T, r, sigma, q, is_call)


def test_batch_empty():
    e = np.empty(0)
    assert qp.eq_bs_price_batch(e, e, e, e, e, e, np.empty(0, bool)).shape == (0,)


# ------------------------------------------------------------- equity VaR


def _pnl(seed, n=750):
    rng = np.random.default_rng(seed)
    return rng.standard_t(4, size=n) * 1e4


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.025, 0.05, 0.2])
@pytest.mark.parametrize("n", [250, 251, 1000])
def test_eq_historical_var_es_match_reference(alpha, n):
    pnl = _pnl(31 + n, n)
    # Order statistic: VaR is one sample value, so it must match exactly.
    assert qp.eq_historical_var(pnl, alpha=alpha) == eq_var.historical_var(pnl, alpha)
    # ES is a short sum over the tail: a different summation order can move
    # it by a few ulps of the tail total; 1e-13 relative covers k <= 250
    # terms (k eps ~ 5.6e-14).
    ref = eq_var.expected_shortfall(pnl, alpha)
    assert qp.eq_expected_shortfall(pnl, alpha=alpha) == pytest.approx(ref, rel=1e-13)


def _cov(seed, n=5):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) * 0.01
    return a @ a.T + np.eye(n) * 1e-5


@pytest.mark.parametrize("dist", ["normal", "t"])
@pytest.mark.parametrize("horizon", [1, 10])
def test_eq_parametric_var_matches_reference(dist, horizon):
    cov = _cov(41)
    w = np.array([1e6, -5e5, 2e5, 3e5, -1e5])
    for alpha in (0.01, 0.05):
        ref = eq_var.parametric_var(w, cov, alpha, dist, 5.0, 100.0, horizon)
        got = qp.eq_parametric_var(w, cov, alpha=alpha, dist=dist, df=5.0, mean=100.0,
                                   horizon_days=horizon)
        # sqrt(w' S w) times a quantile: the quadratic form is a 25-term sum
        # (~25 eps), the inverse CDF is accurate to ~1e-15 (normal) and to
        # ~1e-13 for the engine's own Student-t quantile vs scipy. 1e-12
        # relative covers both with margin.
        assert got == pytest.approx(ref, rel=1e-12)


def test_eq_monte_carlo_var_is_statistically_right():
    cov = _cov(42)
    w = np.array([1e6, -5e5, 2e5, 3e5, -1e5])
    exact = eq_var.parametric_var(w, cov, 0.01)
    got = qp.eq_monte_carlo_var(w, cov, alpha=0.01, n_paths=200_000, seed=5)
    assert got == qp.eq_monte_carlo_var(w, cov, alpha=0.01, n_paths=200_000, seed=5)
    # Quantile SE: sqrt(a(1-a)/n)/phi(z) / z = 0.36 % of VaR at n = 2e5,
    # a = 0.01. 1.5 % is about 4 SE (fixed seed, deterministic).
    assert got == pytest.approx(exact, rel=0.015)


@pytest.mark.parametrize("n_obs,x,alpha", [(250, 0, 0.01), (250, 5, 0.01), (250, 12, 0.01),
                                           (500, 7, 0.01), (1000, 60, 0.05), (250, 250, 0.01),
                                           (1, 1, 0.3)])
def test_eq_backtests_match_reference(n_obs, x, alpha):
    ref = eq_var.kupiec_pof(n_obs=n_obs, n_exceptions=x, alpha=alpha)
    got = qp.eq_kupiec_pof(n_obs=n_obs, n_exceptions=x, alpha=alpha)
    # LR is a difference of two log-likelihood sums (catastrophic only when
    # both are large and close; abs 1e-9 covers n <= 1000); the p-value is a
    # chi2(1) tail from the engine's own erfc vs scipy: 1e-12 relative plus
    # abs 1e-15 for tails near 0.
    assert got["lr"] == pytest.approx(ref["lr"], rel=1e-12, abs=1e-9)
    assert got["pvalue"] == pytest.approx(ref["pvalue"], rel=1e-10, abs=1e-15)
    assert got["expected"] == pytest.approx(ref["expected"], rel=1e-15)
    assert got["rate"] == ref["rate"]
    b_ref = eq_var.basel_traffic_light(n_obs=n_obs, n_exceptions=x, alpha=alpha)
    b = qp.eq_basel_traffic_light(n_obs=n_obs, n_exceptions=x, alpha=alpha)
    assert b["zone"] == b_ref["zone"]
    assert b["multiplier"] == b_ref["multiplier"]
    assert b["plus_factor"] == b_ref["plus_factor"]
    assert b["cumulative_prob"] == pytest.approx(b_ref["cumulative_prob"], rel=1e-12)


# ------------------------------------------------------------------ FX VaR


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.2])
@pytest.mark.parametrize("n", [100, 250, 1001])
def test_fx_empirical_var_es_match_reference(alpha, n):
    pnl = _pnl(51 + n, n)
    var_ref, es_ref = fx_var.empirical_var_es(pnl, alpha)
    var, es = qp.fx_empirical_var_es(pnl, alpha=alpha)
    assert var == var_ref  # order statistic
    assert es == pytest.approx(es_ref, rel=1e-13)  # tail sum, see equity test


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.25])
def test_fx_closed_forms_match_reference(alpha):
    for sigma, mean in ((1e4, 0.0), (2.5e5, -300.0), (0.0, 10.0)):
        # Normal: scipy ndtri vs the engine's inverse normal, both ~1 ulp;
        # t: scipy's t quantile vs the engine's incomplete-beta inversion,
        # agreement ~1e-13 relative. 1e-12 covers both.
        assert qp.fx_normal_var(sigma, alpha=alpha, mean=mean) == pytest.approx(
            fx_var.normal_var(sigma, alpha, mean), rel=1e-12, abs=1e-9)
        assert qp.fx_normal_es(sigma, alpha=alpha, mean=mean) == pytest.approx(
            fx_var.normal_es(sigma, alpha, mean), rel=1e-12, abs=1e-9)
        for df in (3.0, 6.0, 30.0):
            assert qp.fx_t_var(sigma, alpha=alpha, df=df, mean=mean) == pytest.approx(
                fx_var.t_var(sigma, alpha, df, mean), rel=1e-12, abs=1e-9)
            assert qp.fx_t_es(sigma, alpha=alpha, df=df, mean=mean) == pytest.approx(
                fx_var.t_es(sigma, alpha, df, mean), rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("dist", ["normal", "t"])
def test_fx_var_covar_matches_reference(dist):
    import pandas as pd

    cov = _cov(61, 4)
    names = ["EUR", "GBP", "JPY", "CHF"]
    w = np.array([2e6, -1e6, 5e5, 7e5])
    ref = fx_var.var_covar(pd.Series(w, index=names), pd.DataFrame(cov, index=names,
                                                                   columns=names),
                           0.01, 10.0, dist, 5.0, 50.0)
    got = qp.fx_var_covar(w, cov, alpha=0.01, horizon_days=10.0, dist=dist, df=5.0, mean=50.0)
    assert got == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("n_obs,x,alpha", [(250, 0, 0.01), (250, 6, 0.01), (250, 11, 0.01),
                                           (750, 30, 0.025), (250, 250, 0.01)])
def test_fx_backtests_match_reference(n_obs, x, alpha):
    lr_ref, p_ref = fx_var.kupiec_pof(n_obs=n_obs, n_exceptions=x, alpha=alpha)
    lr, p = qp.fx_kupiec_pof(n_obs=n_obs, n_exceptions=x, alpha=alpha)
    assert lr == pytest.approx(lr_ref, rel=1e-12, abs=1e-9)  # see equity test
    assert p == pytest.approx(p_ref, rel=1e-10, abs=1e-15)
    ref = fx_var.basel_traffic_light(n_obs=n_obs, n_exceptions=x, alpha=alpha)
    got = qp.fx_basel_traffic_light(n_obs=n_obs, n_exceptions=x, alpha=alpha)
    assert got["zone"] == ref.zone
    assert got["n_exceptions"] == ref.n_exceptions and got["n_obs"] == ref.n_obs
    assert got["multiplier"] == ref.multiplier and got["plus_factor"] == ref.plus_factor
    assert got["cumulative_prob"] == pytest.approx(ref.cumulative_prob, rel=1e-12)


# -------------------------------------------------------------- error paths

NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("bad", [
    (NAN, 100, 1, 0.05, 0.2), (100, INF, 1, 0.05, 0.2), (100, 100, -1, 0.05, 0.2),
    (100, 100, 1, NAN, 0.2), (100, 100, 1, 0.05, -0.2), (-1, 100, 1, 0.05, 0.2),
])
def test_eq_rejects_bad_inputs(bad):
    S, K, T, r, sigma = bad
    with pytest.raises(ValueError):
        qp.eq_bs_price(S, K, T, r, sigma)
    with pytest.raises(ValueError):
        qp.eq_bs_greeks(S, K, T, r, sigma)
    with pytest.raises(ValueError):
        eq_options.bs_price(S, K, T, r, sigma)  # the reference agrees


def test_eq_greeks_and_iv_domain():
    with pytest.raises(ValueError):
        qp.eq_bs_greeks(100, 100, 0.0, 0.05, 0.2)  # T = 0: Greeks undefined
    with pytest.raises(ValueError):
        qp.eq_implied_vol(10.0, 100, 100, 0.0, 0.05)  # expired
    with pytest.raises(ValueError):
        qp.eq_implied_vol(NAN, 100, 100, 1.0, 0.05)
    with pytest.raises(ValueError):
        qp.eq_implied_vol(200.0, 100, 100, 1.0, 0.05)  # above the upper bound
    with pytest.raises(ValueError):
        qp.eq_crr_price(100, 100, 1, 0.05, 0.2, n_steps=0)
    with pytest.raises(ValueError):
        qp.eq_mc_price(100, 100, 1, 0.05, 0.2, n_paths=1)
    with pytest.raises(ValueError):
        qp.eq_mc_price(100, 100, 1, 0.05, 0.2, threads=0)
    with pytest.raises(ValueError, match="option_type"):
        qp.eq_bs_price(100, 100, 1, 0.05, 0.2, 0.0, "straddle")


def test_fx_rejects_bad_inputs():
    for args in [(NAN, 1.1, 1, 0.05, 0.02, 0.1), (1.1, 1.1, 1, INF, 0.02, 0.1),
                 (1.1, 1.1, 1, 0.05, 0.02, -0.1), (1.1, -1.0, 1, 0.05, 0.02, 0.1)]:
        with pytest.raises(ValueError):
            qp.fx_gk_price(*args, "call")
        with pytest.raises(ValueError):
            fx_options.gk_price(*args, "call")
    with pytest.raises(ValueError):
        qp.fx_implied_vol(0.05, 1.1, 1.1, 0.0, 0.05, 0.02, "call")
    with pytest.raises(ValueError):
        qp.fx_binomial_price(1.1, 1.1, 1, 0.05, 0.02, 0.1, "call", steps=0)
    with pytest.raises(ValueError):
        qp.fx_mc_price(1.1, 1.1, 1, 0.05, 0.02, 0.1, "call", n_paths=2)


@pytest.mark.parametrize("alpha", [0.99, 0.95, 0.5, 0.0, -0.01, NAN, INF])
def test_alpha_is_tail_probability_confidence_rejected(alpha):
    pnl = _pnl(71)
    calls = [
        lambda: qp.eq_historical_var(pnl, alpha=alpha),
        lambda: qp.eq_expected_shortfall(pnl, alpha=alpha),
        lambda: qp.eq_parametric_var(np.ones(2), np.eye(2), alpha=alpha),
        lambda: qp.eq_kupiec_pof(n_obs=250, n_exceptions=3, alpha=alpha),
        lambda: qp.eq_basel_traffic_light(n_obs=250, n_exceptions=3, alpha=alpha),
        lambda: qp.fx_empirical_var_es(pnl, alpha=alpha),
        lambda: qp.fx_normal_var(1.0, alpha=alpha),
        lambda: qp.fx_t_es(1.0, alpha=alpha),
        lambda: qp.fx_var_covar(np.ones(2), np.eye(2), alpha=alpha),
        lambda: qp.fx_kupiec_pof(n_obs=250, n_exceptions=3, alpha=alpha),
        lambda: qp.fx_basel_traffic_light(n_obs=250, n_exceptions=3, alpha=alpha),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_var_rejects_bad_samples_and_counts():
    with pytest.raises(ValueError):
        qp.eq_historical_var(np.array([1.0, NAN] * 200), alpha=0.01)
    with pytest.raises(ValueError):
        qp.fx_empirical_var_es(np.array([]), alpha=0.01)
    with pytest.raises(ValueError):
        qp.fx_empirical_var_es(np.array([1.0, INF]), alpha=0.01)
    with pytest.raises(ValueError):
        qp.eq_parametric_var(np.ones(3), np.eye(2), alpha=0.01)  # shape mismatch
    with pytest.raises(ValueError):
        qp.eq_parametric_var(np.ones(2), np.eye(2), alpha=0.01, dist="cauchy")
    with pytest.raises(ValueError):
        qp.fx_t_var(1.0, alpha=0.01, df=2.0)
    for fn in (qp.eq_kupiec_pof, qp.eq_basel_traffic_light, qp.fx_kupiec_pof,
               qp.fx_basel_traffic_light):
        with pytest.raises(ValueError):
            fn(n_obs=250, n_exceptions=251)
        with pytest.raises(ValueError):
            fn(n_obs=0, n_exceptions=0)
        with pytest.raises(ValueError):
            fn(n_obs=250, n_exceptions=-1)


def test_backtests_are_keyword_only():
    with pytest.raises(TypeError):
        qp.eq_kupiec_pof(250, 3)  # type: ignore[misc]
    with pytest.raises(TypeError):
        qp.fx_basel_traffic_light(250, 3)  # type: ignore[misc]
