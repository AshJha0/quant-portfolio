"""Monte Carlo VaR: simulate factor moves, fully revalue, take the quantile.

Factor models: multivariate normal, or multivariate Student-t (normal /
sqrt(chi2/df) with one common mixing variable, so every factor pair shares the
same df and tail dependence), both matched to a target covariance.

Shock convention for revaluation (spec S10): simulated moves of price
factors are LOG returns, applied as ``S * exp(r)``, so a simulated spot is
always positive. Vol factors move by an absolute change and the implied vol
is floored at zero. No path can make revaluation raise.

Covariance factorisation (spec S7) is ``cholesky``: relative symmetry check,
jitter capped at ``1e-6 * mean(diag)`` with a ``RuntimeWarning``, and optional
explicit ``repair="clip"`` or ``repair="higham"``.

All simulation takes an explicit seed or ``numpy.random.Generator``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np
from qp_core.model import model
from scipy.stats import beta as beta_dist
from scipy.stats import norm

from . import _tail
from ._validation import finite_array, finite_scalar, integer, validate_alpha
from .portfolio import Portfolio

__all__ = [
    "MonteCarloRisk",
    "cholesky",
    "higham_nearest_correlation",
    "monte_carlo_pnl",
    "monte_carlo_risk",
    "monte_carlo_var",
    "repair_covariance",
    "safe_cholesky",
    "simulate_factor_returns",
    "var_confidence_interval",
    "var_standard_error_bootstrap",
    "var_standard_error_kde",
]

JITTER_START = 1e-12
JITTER_CAP = 1e-6
HIGHAM_TOL = 1e-12
HIGHAM_MAX_ITER = 1000


def _as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is not None:
        integer("seed", seed, min_value=0)
    return np.random.default_rng(seed)


def _square_symmetric(cov: object) -> np.ndarray:
    sig = finite_array("covariance", cov, ndim=2, min_size=1)
    if sig.shape[0] != sig.shape[1]:
        raise ValueError(f"covariance must be square, got shape {sig.shape}")
    scale = float(np.abs(sig).max())
    if float(np.abs(sig - sig.T).max()) > 1e-10 * scale:
        raise ValueError("covariance matrix must be symmetric (|A - A'| > 1e-10 max|A|)")
    if np.any(np.diag(sig) < 0.0):
        raise ValueError("covariance has a negative variance on the diagonal")
    return 0.5 * (sig + sig.T)


def _psd_projection(a: np.ndarray, floor: float = 0.0) -> np.ndarray:
    lam, vec = np.linalg.eigh(a)
    lam = np.maximum(lam, floor)
    out = (vec * lam) @ vec.T
    return 0.5 * (out + out.T)


def higham_nearest_correlation(corr: np.ndarray, tol: float = HIGHAM_TOL,
                               max_iter: int = HIGHAM_MAX_ITER) -> np.ndarray:
    """Nearest correlation matrix in Frobenius norm (Higham 2002).

    Alternating projections onto the PSD cone and the unit-diagonal set with
    Dykstra's correction. Stops when successive iterates differ by less than
    ``tol`` (relative Frobenius norm). Raises if not converged in
    ``max_iter`` iterations.
    """
    a = _square_symmetric(corr)
    tol = finite_scalar("tol", tol, min_value=0.0, strict=True)
    max_iter = integer("max_iter", max_iter, min_value=1)
    y = a.copy()
    ds = np.zeros_like(a)
    for _ in range(max_iter):
        r = y - ds
        x = _psd_projection(r)
        ds = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        if np.linalg.norm(y_new - y) <= tol * np.linalg.norm(y_new):
            return y_new
        y = y_new
    raise np.linalg.LinAlgError(f"Higham nearest-correlation did not converge in {max_iter} iterations")


def _jitter_cholesky(sig: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(sig)
    except np.linalg.LinAlgError:
        pass
    scale = float(np.mean(np.diag(sig)))
    if scale <= 0.0:
        raise np.linalg.LinAlgError("covariance has zero mean variance; nothing to factorise")
    eps = JITTER_START * scale
    eye = np.eye(sig.shape[0])
    while eps <= JITTER_CAP * scale * (1.0 + 1e-9):
        try:
            chol = np.linalg.cholesky(sig + eps * eye)
        except np.linalg.LinAlgError:
            eps *= 10.0
            continue
        warnings.warn(
            f"Cholesky needed diagonal jitter {eps:.3g} ({eps / scale:.1e} x mean variance)",
            RuntimeWarning,
            stacklevel=3,
        )
        return chol
    min_eig = float(np.linalg.eigvalsh(sig).min())
    raise np.linalg.LinAlgError(
        f"covariance is not positive definite (most negative eigenvalue {min_eig:.6g}); "
        f"jitter up to {JITTER_CAP:g} x mean variance did not help. Fix the inputs or pass "
        "repair='clip' or repair='higham'"
    )


def _repaired(sig: np.ndarray, repair: str) -> np.ndarray:
    d = np.sqrt(np.diag(sig))
    if np.any(d <= 0.0):
        raise ValueError(f"repair={repair!r} needs strictly positive variances")
    if repair == "clip":
        out = _psd_projection(sig, floor=1e-10 * float(np.mean(np.diag(sig))))
        s = d / np.sqrt(np.diag(out))
        out = out * np.outer(s, s)
    else:
        out = higham_nearest_correlation(sig / np.outer(d, d)) * np.outer(d, d)
    change = float(np.abs(out - sig).max())
    if change > 0.0:
        warnings.warn(
            f"covariance repaired with {repair!r}: max absolute entry change {change:.3g}",
            RuntimeWarning,
            stacklevel=3,
        )
    return out


def repair_covariance(cov: np.ndarray, method: Literal["clip", "higham"]) -> np.ndarray:
    """The repaired matrix ``cholesky(cov, repair=method)`` factorises (spec S7).

    Emits a ``RuntimeWarning`` when the matrix changes.
    """
    if method not in ("clip", "higham"):
        raise ValueError(f"method must be 'clip' or 'higham', got {method!r}")
    return _repaired(_square_symmetric(cov), method)


def cholesky(cov: np.ndarray, repair: Literal["clip", "higham"] | None = None) -> np.ndarray:
    """Lower-triangular ``L`` with ``L L' = cov`` (spec S7).

    * Symmetry is checked relative to ``max|A|`` (1e-10).
    * If plain Cholesky fails, jitter ``eps * I`` is added, ``eps`` from
      ``1e-12 * mean(diag)`` rising x10 up to ``1e-6 * mean(diag)``. Any jitter
      emits a ``RuntimeWarning``. Above the cap, and with ``repair=None``, it
      raises ``LinAlgError`` naming the most negative eigenvalue.
    * ``repair="clip"``: clip eigenvalues at ``1e-10 * mean(diag)``, restore
      the original variances, warn, then factorise.
    * ``repair="higham"``: Higham (2002) nearest correlation with Dykstra's
      correction, rescaled by the original vols, warn, then factorise.
    """
    sig = _square_symmetric(cov)
    if repair is not None:
        if repair not in ("clip", "higham"):
            raise ValueError(f"repair must be None, 'clip' or 'higham', got {repair!r}")
        sig = _repaired(sig, repair)
    return _jitter_cholesky(sig)


def safe_cholesky(cov: np.ndarray, jitter: float = 1e-10, max_tries: int = 12) -> np.ndarray:
    """Deprecated alias of ``cholesky(cov)``.

    The old version escalated jitter to 10 x the mean variance without a
    warning. The alias keeps the name for one release but applies the
    ``cholesky`` cap and warning. ``jitter`` and ``max_tries`` are ignored.
    """
    warnings.warn(
        "safe_cholesky is deprecated; use cholesky(cov, repair=None|'clip'|'higham'). "
        "Jitter is now capped at 1e-6 x mean variance",
        DeprecationWarning,
        stacklevel=2,
    )
    return cholesky(cov)


def simulate_factor_returns(
    cov: np.ndarray,
    n_paths: int,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    mean: np.ndarray | None = None,
    seed: int | np.random.Generator | None = 0,
    repair: Literal["clip", "higham"] | None = None,
) -> np.ndarray:
    """Simulate factor moves with a target covariance, shape ``(n_paths, n)``.

    ``dist="t"``: ``Z / sqrt(W/df)`` with scale ``cov * (df-2)/df`` so the
    covariance matches ``cov`` while the tails fatten. ``repair`` is passed to
    ``cholesky``.
    """
    n_paths = integer("n_paths", n_paths, min_value=1)
    if dist not in ("normal", "t"):
        raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")
    sig = _square_symmetric(cov)
    n = sig.shape[0]
    mu = np.zeros(n) if mean is None else finite_array("mean", mean, ndim=1)
    if mu.size != n:
        raise ValueError(f"mean has {mu.size} entries, covariance is {n}x{n}")
    if dist == "normal":
        chol = cholesky(sig, repair)
        rng = _as_rng(seed)
        z = rng.standard_normal((n_paths, n))
        return mu + z @ chol.T
    df = finite_scalar("df", df)
    if not df > 2.0:
        raise ValueError(f"Student-t df must be > 2 for finite variance, got {df}")
    chol = cholesky(sig * (df - 2.0) / df, repair)
    rng = _as_rng(seed)
    z = rng.standard_normal((n_paths, n))
    w = rng.chisquare(df, size=n_paths) / df
    return mu + (z @ chol.T) / np.sqrt(w)[:, None]


def _log_to_simple(portfolio: Portfolio, moves: np.ndarray) -> np.ndarray:
    """Convert simulated log returns on price factors to the simple returns ``Portfolio.pnl`` takes."""
    out = moves.copy()
    for j, name in enumerate(portfolio.factor_names):
        if portfolio.factors[name].kind != "vol":
            out[:, j] = np.expm1(moves[:, j])
    return out


def monte_carlo_pnl(
    portfolio: Portfolio,
    cov: np.ndarray,
    n_paths: int = 100_000,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    method: Literal["full", "delta_gamma"] = "full",
    seed: int | np.random.Generator | None = 0,
    repair: Literal["clip", "higham"] | None = None,
) -> np.ndarray:
    """Simulate factor scenarios and revalue the portfolio (full revaluation by default).

    ``cov`` is the covariance of daily log returns (price factors) and
    absolute vol changes (vol factors). Raises if a revalued P&L is not finite.
    """
    if not isinstance(portfolio, Portfolio):
        raise ValueError("portfolio must be an eq_var.Portfolio")
    if method not in ("full", "delta_gamma"):
        raise ValueError(f"method must be 'full' or 'delta_gamma', got {method!r}")
    scen = simulate_factor_returns(cov, n_paths, dist=dist, df=df, seed=seed, repair=repair)
    if scen.shape[1] != portfolio.n_factors:
        raise ValueError(f"covariance is {scen.shape[1]}x{scen.shape[1]}, portfolio has "
                         f"{portfolio.n_factors} factors")
    pnl = portfolio.pnl(_log_to_simple(portfolio, scen), method=method)
    if not np.isfinite(pnl).all():
        raise ValueError("revalued Monte Carlo P&L contains NaN or infinite values")
    return pnl


@model(id="EQ-VAR-MC")
def monte_carlo_var(
    portfolio: Portfolio,
    cov: np.ndarray,
    alpha: float = 0.01,
    n_paths: int = 100_000,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    method: Literal["full", "delta_gamma"] = "full",
    seed: int | np.random.Generator | None = 0,
) -> float:
    """Monte Carlo VaR (positive for a loss), S2 order statistic of the revalued P&L."""
    alpha = validate_alpha(alpha)
    pnl = monte_carlo_pnl(portfolio, cov, n_paths, dist, df, method, seed)
    return _tail.var_es(pnl, alpha)[0]


@dataclass(frozen=True)
class MonteCarloRisk:
    """Result of :func:`monte_carlo_risk`: VaR, ES and the bootstrap SE of the
    VaR, all from the one scenario set in ``pnl``.

    Frozen, so a result cannot be edited after the fact. Read the fields as
    attributes (``res.var``); the mapping access of the old dict result
    (``res["var"]``, ``res.keys()``, ``dict(res)``) still works.
    """

    var: float
    es: float
    var_se: float
    pnl: np.ndarray

    def __getitem__(self, key: str) -> object:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)

    def keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(fields(self))


def monte_carlo_risk(
    portfolio: Portfolio,
    cov: np.ndarray,
    alpha: float = 0.01,
    n_paths: int = 100_000,
    dist: Literal["normal", "t"] = "normal",
    df: float = 6.0,
    method: Literal["full", "delta_gamma"] = "full",
    seed: int | np.random.Generator | None = 0,
    n_boot: int = 200,
    repair: Literal["clip", "higham"] | None = None,
) -> MonteCarloRisk:
    """Simulate once and report VaR, ES and the bootstrap VaR standard error (spec S10).

    All three numbers come from the same scenario set. Returns a frozen
    :class:`MonteCarloRisk` with ``var``, ``es``, ``var_se`` (bootstrap,
    primary) and ``pnl``; ``res["var"]`` style access still works.
    """
    alpha = validate_alpha(alpha)
    n_boot = integer("n_boot", n_boot, min_value=2)
    rng = _as_rng(seed)
    pnl = monte_carlo_pnl(portfolio, cov, n_paths, dist, df, method, rng, repair)
    var, es = _tail.var_es(pnl, alpha)
    se = _bootstrap_se(pnl, alpha, n_boot, rng)
    return MonteCarloRisk(var=var, es=es, var_se=se, pnl=pnl)


def _sample(pnl: object, min_size: int) -> np.ndarray:
    arr = finite_array("pnl", pnl, ndim=1, min_size=0)
    if arr.size < min_size:
        raise ValueError(f"need at least {min_size} observations, got {arr.size}")
    return arr


def _bootstrap_se(arr: np.ndarray, alpha: float, n_boot: int, rng: np.random.Generator) -> float:
    # Chunks of at most 50 resamples: memory O(50 n), never an (n_boot, n) matrix.
    chunk = min(n_boot, 50)
    boot = np.empty(n_boot)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, arr.size, size=(m, arr.size))
        boot[done : done + m] = _tail.var_rows(arr[idx], alpha)
        done += m
    return float(np.std(boot, ddof=1))


def var_standard_error_bootstrap(
    pnl: np.ndarray,
    alpha: float = 0.01,
    n_boot: int = 500,
    seed: int | np.random.Generator | None = 0,
) -> float:
    """Bootstrap standard error of the S2 VaR (sample std of resampled VaRs, ddof=1).

    Resampling is chunked. The bootstrap of a sample quantile is lumpy at
    small ``n`` because it can only return order statistics.
    """
    alpha = validate_alpha(alpha)
    n_boot = integer("n_boot", n_boot, min_value=2)
    arr = _sample(pnl, 10)
    return _bootstrap_se(arr, alpha, n_boot, _as_rng(seed))


def var_standard_error_kde(pnl: np.ndarray, alpha: float = 0.01) -> float:
    """Asymptotic VaR standard error ``sqrt(alpha (1-alpha) / n) / f(q)`` (diagnostic).

    ``f`` is a Gaussian kernel density with Silverman's bandwidth, evaluated
    at the reported S2 quantile ``q``. Use the bootstrap as the primary SE.
    """
    alpha = validate_alpha(alpha)
    arr = _sample(pnl, 10)
    n = arr.size
    sd = float(np.std(arr, ddof=1))
    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
    spread = min(sd, iqr / 1.349) if iqr > 0 else sd
    if spread <= 0.0:
        raise ValueError("pnl has zero spread; the density at the quantile is undefined")
    h = 0.9 * spread * n ** (-0.2)
    q = -_tail.var_es(arr, alpha)[0]
    dens = float(np.mean(norm.pdf((q - arr) / h)) / h)
    return float(np.sqrt(alpha * (1.0 - alpha) / n) / dens)


def var_confidence_interval(
    pnl: np.ndarray, alpha: float = 0.01, conf: float = 0.95
) -> tuple[float, float]:
    """Order-statistics (distribution-free) confidence interval for VaR.

    ``P(X_(k) <= q_alpha) = P(Beta(k, n-k+1) <= alpha)``. Picks ranks
    ``(lo, hi)`` with ``P(X_(lo) <= q_alpha <= X_(hi)) >= conf`` and returns
    ``(-X_(hi), -X_(lo))`` as (lower, upper) positive losses. Raises when the
    sample is too short for any rank to give the requested coverage.
    """
    alpha = validate_alpha(alpha)
    conf = finite_scalar("conf", conf)
    if not 0.5 < conf < 1.0:
        raise ValueError(f"conf must be in (0.5, 1), got {conf}")
    arr = np.sort(_sample(pnl, 20))
    n = arr.size
    tail = (1.0 - conf) / 2.0
    ranks = np.arange(1, n + 1)
    cdf_at_alpha = beta_dist.cdf(alpha, ranks, n - ranks + 1)
    lo_candidates = np.flatnonzero(cdf_at_alpha >= 1.0 - tail)
    hi_candidates = np.flatnonzero(cdf_at_alpha <= tail)
    if lo_candidates.size == 0 or hi_candidates.size == 0:
        raise ValueError(
            f"n={n} is too short for a {conf:.0%} order-statistic interval at alpha={alpha}"
        )
    return float(-arr[int(hi_candidates[0])]), float(-arr[int(lo_candidates[-1])])
