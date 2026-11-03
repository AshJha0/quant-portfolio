"""Almgren-Chriss optimal execution, discrete time, from scratch.

Setup (Almgren & Chriss 2001, "Optimal execution of portfolio transactions",
*Journal of Risk* 3(2), 5-39):
sell/buy ``X`` shares over ``N`` intervals of length ``tau = T/N``.  Holdings
``x_0 = X, ..., x_N = 0``; trades ``n_k = x_{k-1} - x_k``.  Arithmetic price
dynamics with volatility ``sigma`` (currency / sqrt(time)), **linear
permanent impact** ``gamma`` (currency per share) and **linear temporary
impact** ``eta`` (currency per share per unit trade rate), fixed cost
``epsilon`` per share:

    E[cost] = 0.5*gamma*X^2 + epsilon*sum|n_k| + (eta_tilde/tau) * sum n_k^2
    V[cost] = sigma^2 * tau * sum_{k=1..N} x_k^2,   eta_tilde = eta - gamma*tau/2

Minimising ``E + lambda * V`` (risk aversion ``lambda >= 0``) gives the
discrete Euler-Lagrange recursion

    x_{k-1} - 2*cosh(kappa*tau)*x_k + x_{k+1} = 0

whose solution with the boundary conditions is the closed form

    x_j = X * sinh(kappa * (T - t_j)) / sinh(kappa * T),   t_j = j*tau,

where ``kappa`` solves ``cosh(kappa*tau) = 1 + (kappa_tilde^2 * tau^2)/2``
and ``kappa_tilde^2 = lambda * sigma^2 / eta_tilde``.

Limits: ``lambda -> 0`` gives equal slices (TWAP); ``lambda -> inf`` dumps
everything in the first slice (urgency).  Both are tested.

Power-law temporary impact.  The simulator in :mod:`eq_algo.intraday`
charges ``temp_coef * sigma * (n_k / V_k)^delta`` per share, a square root
by default (Almgren, Thum, Hauptmann and Li 2005).  Under that law the
cost of slice ``k`` is ``c_k * n_k^(1+delta)`` with
``c_k = S * temp_coef * sigma / V_k^delta``, and there is no closed form.
:func:`power_law_trajectory` minimises

    sum_k c_k * |n_k|^(1+delta) + lambda * sum_k s2_k * x_k^2

to machine precision by Newton's method.  The problem is a chain: slice
``k`` couples only ``x_{k-1}`` and ``x_k``, so the Hessian is tridiagonal
and each Newton step is one backward elimination and one forward
substitution, the dynamic-programming (Riccati) sweep of the local
quadratic model.  With ``delta = 1`` the problem is quadratic, one step is
exact, and the result equals the closed form above to 1e-8 of ``X``
(tested).  :func:`compare_impact_laws` scores the linear-law and the
power-law optimum under both laws side by side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .intraday import IntradayMarket

__all__ = [
    "ACParams",
    "ac_cost_moments",
    "ac_kappa",
    "ac_trades",
    "ac_trajectory",
    "compare_impact_laws",
    "efficient_frontier",
    "evaluate_schedules",
    "power_law_objective",
    "power_law_trajectory",
]


@dataclass(frozen=True)
class ACParams:
    """Almgren-Chriss model parameters (arithmetic, absolute units).

    ``sigma``: price vol in currency per sqrt(day); ``eta``: temporary impact
    (currency per (share/day)); ``gamma``: permanent impact (currency per
    share); ``epsilon``: fixed cost per share (half-spread + fees);
    ``total_time``: horizon in days; ``n_slices``: number of intervals.
    """

    total_shares: float
    n_slices: int
    total_time: float = 1.0
    sigma: float = 1.0
    eta: float = 1e-6
    gamma: float = 0.0
    epsilon: float = 0.0

    def __post_init__(self) -> None:
        if self.total_shares <= 0:
            raise ValueError("total_shares must be > 0")
        if self.n_slices < 1:
            raise ValueError("n_slices must be >= 1")
        if self.total_time <= 0:
            raise ValueError("total_time must be > 0")
        if self.sigma < 0 or self.eta <= 0 or self.gamma < 0 or self.epsilon < 0:
            raise ValueError("require sigma, gamma, epsilon >= 0 and eta > 0")
        if self.eta_tilde <= 0:
            raise ValueError(
                "eta_tilde = eta - gamma*tau/2 must be > 0 "
                "(temporary impact too small vs permanent at this slicing)"
            )

    @property
    def tau(self) -> float:
        return self.total_time / self.n_slices

    @property
    def eta_tilde(self) -> float:
        return self.eta - 0.5 * self.gamma * self.tau


def ac_kappa(params: ACParams, lam: float) -> float:
    """Urgency parameter ``kappa`` (1/time) for risk aversion ``lam >= 0``.

    Solves ``cosh(kappa*tau) = 1 + kappa_tilde^2 * tau^2 / 2`` exactly via
    ``arccosh``; returns 0 for ``lam = 0`` (TWAP limit).
    """
    if lam < 0:
        raise ValueError("lambda (risk aversion) must be >= 0")
    if lam == 0:
        return 0.0
    kt2 = lam * params.sigma**2 / params.eta_tilde
    tau = params.tau
    return float(np.arccosh(1.0 + 0.5 * kt2 * tau**2) / tau)


def _sinh_ratio(a: np.ndarray, b: float) -> np.ndarray:
    """``sinh(a) / sinh(b)`` for ``0 <= a <= b``, overflow-free.

    The naive form overflows for ``b`` beyond ~710 (``sinh`` exceeds the
    double range) and returns ``inf/inf = NaN``.  Rewriting

        sinh(a)/sinh(b) = e^{a-b} * (1 - e^{-2a}) / (1 - e^{-2b})

    keeps every exponential argument non-positive, so the result is finite
    for arbitrarily large urgency; ``expm1`` preserves accuracy as
    ``a, b -> 0``.
    """
    a = np.asarray(a, dtype=float)
    if b <= 0.0:
        raise ValueError("b must be > 0 in _sinh_ratio")
    # -expm1(-2x) == 1 - e^{-2x}; the two minus signs cancel in the ratio.
    return np.exp(a - b) * np.expm1(-2.0 * a) / np.expm1(-2.0 * b)


def ac_trajectory(params: ACParams, lam: float) -> np.ndarray:
    """Optimal holdings ``x_0..x_N`` (length ``n_slices + 1``).

    ``x_j = X * sinh(kappa*(T - t_j)) / sinh(kappa*T)``; exactly linear
    (TWAP) when ``lam = 0``.  Monotone decreasing from ``X`` to 0.

    The sinh ratio is evaluated in an overflow-free exponential form, so
    extreme risk aversion (``kappa*T`` in the hundreds or thousands) returns
    the correct front-loaded schedule instead of NaN.
    """
    X, N, T = params.total_shares, params.n_slices, params.total_time
    t = np.arange(N + 1) * params.tau
    kappa = ac_kappa(params, lam)
    if kappa == 0.0:
        return X * (1.0 - t / T)
    x = X * _sinh_ratio(kappa * np.maximum(T - t, 0.0), kappa * T)
    x[0] = X
    x[-1] = 0.0
    return x


def ac_trades(params: ACParams, lam: float) -> np.ndarray:
    """Trade list ``n_k = x_{k-1} - x_k`` (length ``n_slices``); sums to X."""
    x = ac_trajectory(params, lam)
    return -np.diff(x)


def ac_cost_moments(params: ACParams, trajectory: np.ndarray) -> tuple[float, float]:
    """Expected cost and variance of cost for an arbitrary trajectory.

    ``E = 0.5*gamma*X^2 + epsilon*sum|n_k| + (eta_tilde/tau)*sum n_k^2``;
    ``V = sigma^2 * tau * sum_{k=1..N} x_k^2``.  Works for any admissible
    trajectory (x_0 = X, x_N = 0), not just the optimum.
    """
    x = np.asarray(trajectory, dtype=float)
    if x.shape != (params.n_slices + 1,):
        raise ValueError("trajectory must have length n_slices + 1")
    if not np.isclose(x[0], params.total_shares) or not np.isclose(x[-1], 0.0):
        raise ValueError("trajectory must run from X to 0")
    n = -np.diff(x)
    exp_cost = (0.5 * params.gamma * params.total_shares**2
                + params.epsilon * np.abs(n).sum()
                + params.eta_tilde / params.tau * (n**2).sum())
    var_cost = params.sigma**2 * params.tau * (x[1:] ** 2).sum()
    return float(exp_cost), float(var_cost)


def efficient_frontier(params: ACParams, lams: Sequence[float]) -> pd.DataFrame:
    """Expected-cost / variance frontier across risk aversions.

    Columns: ``kappa, expected_cost, variance, std``; indexed by lambda.
    Expected cost is increasing and variance decreasing in lambda (tested).
    """
    rows = []
    for lam in lams:
        x = ac_trajectory(params, lam)
        e, v = ac_cost_moments(params, x)
        rows.append({"lam": float(lam), "kappa": ac_kappa(params, lam),
                     "expected_cost": e, "variance": v, "std": np.sqrt(v)})
    return pd.DataFrame(rows).set_index("lam")


def evaluate_schedules(market: IntradayMarket,
                       schedules: Mapping[str, np.ndarray], side: int = 1,
                       n_reps: int = 200, seed: int = 0) -> pd.DataFrame:
    """Monte Carlo horse race of execution schedules on the simulator.

    Each replication draws one market day (price path + volumes) per seed and
    executes every schedule against the *same* seed (common random numbers,
    variance-reduced comparison).  Reports mean and std of implementation
    shortfall (bps vs the arrival mid) and slippage vs VWAP/TWAP.

    Returns a DataFrame indexed by strategy name with columns
    ``model_implied_is_bps`` (implementation shortfall on the same market
    with zero price and volume noise: the cost the *model* implies, not an
    estimate of what a real market would charge), ``expected_is_bps`` (the
    same numbers under the v2.0.0 name, kept for one release), ``mean_is_bps``
    and ``se_mean_is_bps`` (Monte Carlo mean and its standard error),
    ``std_is_bps``, ``mean_vs_vwap_bps``, ``mean_vs_twap_bps``.

    Publish ``model_implied_is_bps`` for the cost ranking and the Monte Carlo
    columns for the risk: with a few hundred replications the common price
    noise (std of order 100 bp) swamps differences of a few bp in the mean.
    """
    from .benchmarks import benchmark_slippage  # local import avoids cycles

    if n_reps < 2:
        raise ValueError("n_reps must be >= 2")
    names = list(schedules.keys())
    is_bps = {k: np.empty(n_reps) for k in names}
    vw = {k: np.empty(n_reps) for k in names}
    tw = {k: np.empty(n_reps) for k in names}
    for r in range(n_reps):
        for k in names:
            res = market.execute(np.asarray(schedules[k], dtype=float), side=side,
                                 seed=seed + r)
            s = benchmark_slippage(res)
            is_bps[k][r] = s["vs_arrival_bps"]
            vw[k][r] = s["vs_vwap_bps"]
            tw[k][r] = s["vs_twap_bps"]
    from dataclasses import replace as _replace

    quiet = IntradayMarket(_replace(market.config, price_noise=0.0, vol_noise=0.0))
    expected = {}
    for k in names:
        res0 = quiet.execute(np.asarray(schedules[k], dtype=float), side=side, seed=0)
        expected[k] = benchmark_slippage(res0)["vs_arrival_bps"]
    rows = {k: {
        "model_implied_is_bps": expected[k],
        "expected_is_bps": expected[k],
        "mean_is_bps": is_bps[k].mean(),
        "se_mean_is_bps": is_bps[k].std(ddof=1) / np.sqrt(n_reps),
        "std_is_bps": is_bps[k].std(ddof=1),
        "mean_vs_vwap_bps": vw[k].mean(),
        "mean_vs_twap_bps": tw[k].mean(),
    } for k in names}
    return pd.DataFrame.from_dict(rows, orient="index")


def _thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray,
            rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system by elimination and back substitution."""
    n = diag.size
    c = np.empty(n)
    d = np.empty(n)
    c[0] = upper[0] / diag[0] if n > 1 else 0.0
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / m
    out = np.empty(n)
    out[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        out[i] = d[i] - c[i] * out[i + 1]
    return out


def _power_law_inputs(cost_coeffs, delta, var_per_step, n):
    c = np.asarray(cost_coeffs, dtype=float)
    if c.shape != (n,) or not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("cost_coeffs must be finite and > 0, one per slice")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be > 0 (0.5 square root, 1 linear)")
    s2 = np.broadcast_to(np.asarray(var_per_step, dtype=float), (n,)).astype(float)
    if not np.all(np.isfinite(s2)) or np.any(s2 < 0):
        raise ValueError("var_per_step must be finite and >= 0")
    return c, s2


def power_law_objective(trajectory: np.ndarray, cost_coeffs: np.ndarray,
                        delta: float, risk_aversion: float,
                        var_per_step: float | np.ndarray) -> float:
    """``sum c_k |n_k|^(1+delta) + lambda * sum_{k=1..N} s2_k x_k^2``.

    ``trajectory`` is holdings ``x_0..x_N``; ``cost_coeffs`` has length N and
    is in currency per share^(1+delta); ``var_per_step`` is the price
    variance per slice in currency^2 (scalar or length N, applied to
    ``x_1..x_N``); ``risk_aversion`` is in 1/currency.
    """
    x = np.asarray(trajectory, dtype=float)
    c, s2 = _power_law_inputs(cost_coeffs, delta, var_per_step, x.size - 1)
    n = -np.diff(x)
    return float((c * np.abs(n) ** (1.0 + delta)).sum()
                 + risk_aversion * (s2 * x[1:] ** 2).sum())


def power_law_trajectory(total_shares: float, cost_coeffs: np.ndarray,
                         delta: float, risk_aversion: float,
                         var_per_step: float | np.ndarray,
                         tol: float = 1e-13, max_iter: int = 200) -> np.ndarray:
    """Mean-variance optimal holdings under power-law temporary impact.

    Minimises :func:`power_law_objective` over ``x_1..x_{N-1}`` with
    ``x_0 = X`` and ``x_N = 0``.  The objective is strictly convex for
    ``delta > 0``, so the stationary point is the global minimum.  Damped
    Newton steps, each solved by a tridiagonal sweep, start from TWAP and
    stop when the step is below ``tol * X``.

    Returns holdings of length ``N + 1``.  Raises ``RuntimeError`` if
    Newton has not converged after ``max_iter`` steps.
    """
    if not np.isfinite(total_shares) or total_shares <= 0:
        raise ValueError("total_shares must be finite and > 0")
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("risk_aversion must be finite and >= 0")
    n_slices = np.asarray(cost_coeffs).size
    c, s2 = _power_law_inputs(cost_coeffs, delta, var_per_step, n_slices)
    X = float(total_shares)
    x = X * (1.0 - np.arange(n_slices + 1) / n_slices)
    if n_slices == 1:
        return x
    floor = 1e-12 * X
    for _ in range(max_iter):
        n = -np.diff(x)
        a = np.maximum(np.abs(n), floor)
        g_n = (1.0 + delta) * c * np.sign(n) * np.abs(n) ** delta     # dF/dn_k
        h = delta * (1.0 + delta) * c * a ** (delta - 1.0)             # d2F/dn_k^2
        s2_in = s2[: n_slices - 1]                                      # x_1..x_{N-1}
        grad = -g_n[:-1] + g_n[1:] + 2.0 * risk_aversion * s2_in * x[1:-1]
        diag = h[:-1] + h[1:] + 2.0 * risk_aversion * s2_in
        off = -h[1:-1]
        step = _thomas(off, diag, off, -grad)
        f0 = power_law_objective(x, c, delta, risk_aversion, s2)
        t = 1.0
        while True:
            trial = x.copy()
            trial[1:-1] += t * step
            if power_law_objective(trial, c, delta, risk_aversion, s2) <= f0 or t < 1e-12:
                break
            t *= 0.5
        x = trial
        if np.max(np.abs(t * step)) <= tol * X:
            return x
    raise RuntimeError("power_law_trajectory: Newton did not converge")


def compare_impact_laws(config, parent_qty: float, risk_aversion: float,
                        side: int = 1) -> pd.DataFrame:
    """Linear-law and power-law optimal schedules, each scored under both laws.

    Both schedules use the forecast volume curve ``profile * day_volume`` and
    the same ``temp_coef``.  ``ac_linear`` is optimal for exponent 1 (the
    Almgren-Chriss law with bucket-specific liquidity) and ``power_law`` for
    ``config.impact_exponent``.  The coefficient is not re-fitted between
    laws, so compare rows within a column, not across columns.  Columns:

    - ``objective_linear_law`` / ``objective_power_law``: the mean-variance
      objective in currency (temporary impact plus ``lambda`` times the
      variance), ``risk_aversion`` in 1/currency;
    - ``cost_linear_law_bps`` / ``cost_power_law_bps``: model-implied
      implementation shortfall on the noise-free simulator (spread,
      temporary and permanent impact) under each exponent.
    """
    from dataclasses import replace as _replace

    from .benchmarks import benchmark_slippage

    delta_p = float(config.impact_exponent)
    vols = config.profile * config.day_volume
    # price variance per bucket in currency^2, from sigma_daily (not from
    # price_noise, which tests set to 0 to make the mid path deterministic)
    s2 = (config.mid0 * config.sigma_daily) ** 2 / config.n_buckets

    def coeffs(delta):
        return config.mid0 * config.temp_coef * config.sigma_daily / vols ** delta

    sched = {
        "ac_linear": power_law_trajectory(parent_qty, coeffs(1.0), 1.0, risk_aversion, s2),
        "power_law": power_law_trajectory(parent_qty, coeffs(delta_p), delta_p,
                                          risk_aversion, s2),
    }
    rows = {}
    for name, x in sched.items():
        row = {}
        for label, d in (("linear_law", 1.0), ("power_law", delta_p)):
            row[f"objective_{label}"] = power_law_objective(x, coeffs(d), d, risk_aversion, s2)
            quiet = IntradayMarket(_replace(config, price_noise=0.0, vol_noise=0.0,
                                            impact_exponent=d))
            res = quiet.execute(np.maximum(-np.diff(x), 0.0), side=side, seed=0)
            row[f"cost_{label}_bps"] = benchmark_slippage(res)["vs_arrival_bps"]
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")
