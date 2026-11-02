"""Monte Carlo VaR: multivariate normal / Student-t / jump-mixture factors.

Factor returns are simulated from a daily covariance matrix and the book is
fully revalued scenario by scenario (options via Garman-Kohlhagen, forwards
via their deposit legs).  The revaluation is vectorised.

Distributions
-------------
* ``"normal"`` - MVN(0, Sigma h).
* ``"t"``      - multivariate Student-t scaled to match Sigma exactly
  (``X = Z sqrt((df-2)/df) / sqrt(W/df)``), so the comparison with normal
  MC is at equal covariance.  One common chi-square mixing variable is
  used, which gives every factor pair the same tail dependence.
* ``"jump"``   - normal diffusion plus a compound-Poisson common jump.  The
  per-day jump probability ``p`` becomes the intensity
  ``lam = -ln(1 - p)``, the jump count over ``h`` days is
  ``N ~ Poisson(lam h)``, and the jump return of each factor is the sum of
  ``N`` iid ``N(mu_J, sigma_J)`` sizes.  The jump is not compensated: it
  adds a drift of ``lam h mu_J`` on top of the zero-mean diffusion.

Cholesky
--------
:func:`robust_cholesky` checks symmetry with a relative tolerance, tries a
plain Cholesky, then adds diagonal jitter from ``1e-12 * mean(diag)`` up to
a cap of ``1e-6 * mean(diag)``.  Any jitter raises a
:class:`~fx_var.common.NumericalWarning`.  Past the cap it raises, naming
the most negative eigenvalue, unless ``repair="clip"`` or
``repair="higham"`` asks for an explicit nearest-PSD repair.

Standard error
--------------
The reported ``se_var`` is the bootstrap standard error of the order-
statistic VaR, computed from the same scenarios as the VaR.
:func:`var_standard_error` (asymptotic formula with a Silverman-bandwidth
Gaussian KDE evaluated at the reported quantile) is a diagnostic.  A
fixed-bandwidth density estimate is biased at deep tails and small
scenario counts, so the bootstrap is the primary figure.
"""

from __future__ import annotations

import warnings as _w
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .book import Book, Market
from .common import (
    NumericalWarning,
    require_positions,
    resolve_alpha,
    validate_finite,
    validate_horizon,
)
from .expected_shortfall import _tail, empirical_var_es, tail_rank

__all__ = [
    "JumpSpec",
    "MonteCarloVaRResult",
    "monte_carlo_var",
    "nearest_correlation_higham",
    "robust_cholesky",
    "simulate_factor_returns",
    "var_standard_error",
    "var_standard_error_bootstrap",
]

_JITTER_START = 1e-12
_JITTER_CAP = 1e-6
_SYM_RTOL = 1e-10


@dataclass(frozen=True)
class JumpSpec:
    """Common-jump overlay for the ``"jump"`` distribution.

    Parameters
    ----------
    prob : float
        Per-day probability of at least one jump event, in [0, 1).  Over a
        horizon of ``h`` days the jump count is Poisson with mean
        ``-ln(1 - prob) * h``.
    mean : mapping factor -> float
        Mean size of one jump per factor (log-return for FX factors).
        Factors not listed do not jump.  E.g. ``{"FX:TRY": -0.15}``.
    std : mapping factor -> float
        Dispersion of one jump per factor (0 allowed).
    """

    prob: float
    mean: Mapping[str, float]
    std: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_finite(prob=self.prob)
        if not (0.0 <= self.prob < 1.0):
            raise ValueError(f"jump prob must be in [0, 1), got {self.prob}")
        for k, v in self.mean.items():
            validate_finite(**{f"jump mean for {k}": v})
        for k, v in self.std.items():
            validate_finite(**{f"jump std for {k}": v})
            if v < 0:
                raise ValueError(f"jump std for {k} must be >= 0, got {v}")

    @property
    def intensity(self) -> float:
        """Poisson intensity per day, ``-ln(1 - prob)``."""
        return float(-np.log1p(-self.prob))


@dataclass(frozen=True)
class MonteCarloVaRResult:
    """Monte Carlo VaR result (positive base-ccy losses).

    ``se_var`` is the bootstrap standard error of ``var`` from the same
    scenarios.  ``alpha`` is the tail probability.
    """

    var: float
    es: float
    alpha: float
    horizon_days: float
    dist: str
    n_scenarios: int
    se_var: float
    pnl: np.ndarray


def _psd_project(a: np.ndarray) -> np.ndarray:
    lam, v = np.linalg.eigh((a + a.T) / 2.0)
    return (v * np.maximum(lam, 0.0)) @ v.T


def nearest_correlation_higham(corr: np.ndarray, tol: float = 1e-12,
                               max_iter: int = 1000) -> np.ndarray:
    """Nearest correlation matrix (Higham 2002), Dykstra-corrected projections.

    Alternates the projection onto the PSD cone and onto the unit-diagonal
    set, with Dykstra's correction on the PSD step.  Stops when successive
    iterates change by less than ``tol`` in relative Frobenius norm.

    Raises
    ------
    ValueError
        If not converged after ``max_iter`` iterations, or input not finite.
    """
    y = np.array(corr, dtype=float)
    if y.ndim != 2 or y.shape[0] != y.shape[1] or not np.isfinite(y).all():
        raise ValueError("corr must be a finite square matrix")
    ds = np.zeros_like(y)
    for _ in range(int(max_iter)):
        r = y - ds
        x = _psd_project(r)
        ds = x - r
        y_new = x.copy()
        np.fill_diagonal(y_new, 1.0)
        diff = float(np.linalg.norm(y_new - y)) / max(float(np.linalg.norm(y_new)), 1e-300)
        y = y_new
        if diff < tol:
            return (y + y.T) / 2.0
    raise ValueError(f"Higham nearest-correlation did not converge in {max_iter} iterations")


def _repair(a: np.ndarray, repair: str) -> np.ndarray:
    d = np.diag(a).copy()
    if np.any(d <= 0):
        raise ValueError("cannot repair a covariance with non-positive variances")
    if repair == "clip":
        lam, v = np.linalg.eigh(a)
        floor = 1e-10 * float(np.mean(d))
        b = (v * np.maximum(lam, floor)) @ v.T
        scale = np.sqrt(d / np.diag(b))
        b = b * np.outer(scale, scale)
    elif repair == "higham":
        sd = np.sqrt(d)
        corr = a / np.outer(sd, sd)
        b = nearest_correlation_higham(corr) * np.outer(sd, sd)
    else:
        raise ValueError(f"repair must be None, 'clip' or 'higham', got {repair!r}")
    _w.warn(f"covariance repaired with method {repair!r} (nearest PSD); "
            "variances kept", NumericalWarning, stacklevel=3)
    return (b + b.T) / 2.0


def _jittered_cholesky(a: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        pass
    base = float(np.mean(np.diag(a)))
    base = base if base > 0 else 1.0
    eps = _JITTER_START * base
    eye = np.eye(a.shape[0])
    while eps <= _JITTER_CAP * base * (1.0 + 1e-9):
        try:
            chol = np.linalg.cholesky(a + eps * eye)
        except np.linalg.LinAlgError:
            eps *= 10.0
            continue
        _w.warn(
            f"covariance not positive definite; Cholesky computed with "
            f"diagonal jitter {eps:.2e} ({eps / base:.0e} x mean variance)",
            NumericalWarning,
            stacklevel=3,
        )
        return chol
    return None


def robust_cholesky(cov: np.ndarray, max_tries: int | None = None,
                    repair: str | None = None) -> np.ndarray:
    """Cholesky factor of ``cov`` with capped jitter and optional repair.

    Parameters
    ----------
    cov : array_like
        Symmetric covariance.  Symmetry is checked relatively:
        ``|A - A'| <= 1e-10 max|A|``.
    max_tries : int, optional
        Deprecated and ignored.  The jitter schedule is fixed: from
        ``1e-12 * mean(diag)``, x10 per try, capped at ``1e-6 * mean(diag)``.
    repair : {None, "clip", "higham"}
        ``None``: raise past the jitter cap.  ``"clip"``: clip eigenvalues
        to ``1e-10 * mean(diag)`` and restore the original variances.
        ``"higham"``: nearest correlation matrix (Higham 2002) rescaled by
        the original volatilities.  A repair always warns.

    Raises
    ------
    ValueError
        Non-square, non-finite or asymmetric input, or not factorisable
        within the jitter cap and ``repair`` is None.
    """
    if max_tries is not None:
        _w.warn("robust_cholesky(max_tries=...) is deprecated and ignored; the "
                "jitter cap is 1e-6 * mean(diag)", DeprecationWarning, stacklevel=2)
    a = np.asarray(cov, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("cov must be a non-empty square matrix")
    if not np.isfinite(a).all():
        raise ValueError(
            "cov contains NaN or infinite entries (NaN policy: refuse, "
            "never impute)"
        )
    amax = float(np.max(np.abs(a)))
    if float(np.max(np.abs(a - a.T))) > _SYM_RTOL * amax:
        raise ValueError("cov must be symmetric (relative tolerance 1e-10)")
    if repair is not None and repair not in ("clip", "higham"):
        raise ValueError(f"repair must be None, 'clip' or 'higham', got {repair!r}")
    chol = _jittered_cholesky(a)
    if chol is not None:
        return chol
    if repair is None:
        lam_min = float(np.linalg.eigvalsh((a + a.T) / 2.0)[0])
        raise ValueError(
            "covariance matrix is not positive definite within the jitter cap "
            f"1e-6 x mean variance (most negative eigenvalue {lam_min:.3e}); "
            "pass repair='clip' or repair='higham' to repair it explicitly"
        )
    b = _repair(a, repair)
    chol = _jittered_cholesky(b)
    if chol is None:
        raise ValueError("repaired covariance still not factorisable")
    return chol


def _as_rng(rng_or_seed) -> np.random.Generator:
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed
    return np.random.default_rng(rng_or_seed)


def simulate_factor_returns(
    cov: pd.DataFrame,
    n_scenarios: int,
    dist: str = "normal",
    df: float = 6.0,
    jumps: JumpSpec | None = None,
    seed: int | np.random.Generator = 0,
    horizon_days: float = 1.0,
    repair: str | None = None,
) -> pd.DataFrame:
    """Simulate factor-return scenarios from a daily covariance.

    Parameters
    ----------
    cov : pandas.DataFrame
        Daily factor covariance (columns = factor names).
    n_scenarios : int
        Number of scenarios (rows of the result).
    dist : {"normal", "t", "jump"}
        See module docstring.  ``"jump"`` requires ``jumps``.
    df : float
        Degrees of freedom for ``"t"`` (must be > 2; covariance-matched).
    jumps : JumpSpec, optional
        Common-jump overlay for ``"jump"``.
    seed : int or numpy.random.Generator
        Reproducibility (explicit by convention).
    horizon_days : float
        Scales the covariance by h and the jump intensity by h.
    repair : {None, "clip", "higham"}
        Passed to :func:`robust_cholesky`.

    Returns
    -------
    pandas.DataFrame
        ``n_scenarios x n_factors`` scenario matrix.
    """
    validate_horizon(horizon_days)
    if int(n_scenarios) != n_scenarios or n_scenarios < 1:
        raise ValueError("n_scenarios must be a positive integer")
    n_scenarios = int(n_scenarios)
    if dist not in ("normal", "t", "jump"):
        raise ValueError(f"dist must be 'normal', 't' or 'jump', got {dist!r}")
    if dist == "t":
        validate_finite(df=df)
        if df <= 2:
            raise ValueError("Student-t df must be > 2 for finite variance")
    if dist == "jump" and jumps is None:
        raise ValueError("dist='jump' requires a JumpSpec")
    rng = _as_rng(seed)
    factors = list(cov.columns)
    a = cov.to_numpy(dtype=float) * horizon_days
    chol = robust_cholesky(a, repair=repair)
    z = rng.standard_normal((n_scenarios, len(factors)))
    x = z @ chol.T
    if dist == "t":
        w = rng.chisquare(df, size=n_scenarios) / df
        x = x * np.sqrt((df - 2.0) / df) / np.sqrt(w)[:, None]
    elif dist == "jump":
        assert jumps is not None
        n_jumps = rng.poisson(jumps.intensity * horizon_days, size=n_scenarios)
        for j, f in enumerate(factors):
            if f in jumps.mean:
                # sum of N iid N(mu, s^2) sizes = N mu + sqrt(N) s Z
                zj = rng.standard_normal(n_scenarios)
                x[:, j] += (n_jumps * jumps.mean[f]
                            + np.sqrt(n_jumps) * jumps.std.get(f, 0.0) * zj)
    return pd.DataFrame(x, columns=factors)


def _check_pnl(pnl, min_n: int = 10) -> np.ndarray:
    arr = np.asarray(pnl, dtype=float).ravel()
    if arr.size < min_n:
        raise ValueError(f"need at least {min_n} scenarios, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError("pnl contains NaN or infinite values (NaN policy: refuse)")
    return arr


def var_standard_error(pnl: np.ndarray, alpha: float | None = None, *,
                       confidence: float | None = None) -> float:
    """Diagnostic asymptotic standard error of the empirical VaR.

    ``SE = sqrt(alpha (1-alpha) / n) / f_hat(q)``, where ``q`` is the
    reported order-statistic quantile (minus the VaR) and ``f_hat`` a
    Gaussian KDE with Silverman's bandwidth
    ``h = 0.9 min(sd, IQR/1.34) n^(-1/5)``.  Prefer
    :func:`var_standard_error_bootstrap`, which is what
    :func:`monte_carlo_var` reports.
    """
    alpha = resolve_alpha(alpha, confidence)
    arr = _check_pnl(pnl)
    n = arr.size
    losses_d, _, idx = _tail(arr, alpha, None)
    q = -float(losses_d[idx])
    sd = float(np.std(arr, ddof=1))
    iqr = float(np.subtract(*np.percentile(arr, [75, 25])))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    if spread <= 0:
        raise ValueError("pnl has zero dispersion; VaR standard error undefined")
    h = 0.9 * spread * n ** (-0.2)
    dens = 0.0
    chunk = 1 << 16
    for i in range(0, n, chunk):
        u = (q - arr[i:i + chunk]) / h
        dens += float(np.sum(np.exp(-0.5 * u * u)))
    dens /= n * h * np.sqrt(2.0 * np.pi)
    dens = max(dens, 1e-300)
    return float(np.sqrt(alpha * (1.0 - alpha) / n) / dens)


def var_standard_error_bootstrap(
    pnl: np.ndarray,
    alpha: float | None = None,
    n_boot: int = 500,
    seed: int | np.random.Generator = 0,
    *,
    confidence: float | None = None,
) -> float:
    """Bootstrap standard error of the empirical (order-statistic) VaR.

    Resamples ``pnl`` with replacement ``n_boot`` times, recomputes the VaR
    of each resample with the same rule as
    :func:`fx_var.expected_shortfall.empirical_var`, and returns the sample
    standard deviation (ddof=1).  Resamples are drawn in chunks of at most
    one at a time, so memory is O(n) and no ``n_boot x n`` index matrix
    is ever built.
    """
    alpha = resolve_alpha(alpha, confidence)
    if int(n_boot) != n_boot or n_boot < 2:
        raise ValueError(f"n_boot must be an integer >= 2, got {n_boot}")
    n_boot = int(n_boot)
    arr = _check_pnl(pnl)
    rng = _as_rng(seed)
    n = arr.size
    m = tail_rank(n, alpha)
    kth = n - m  # ascending position of the m-th largest loss
    losses = -arr
    out = np.empty(n_boot)
    for i in range(n_boot):  # one resample at a time: O(n) memory
        sample = losses[rng.integers(0, n, size=n)]
        out[i] = np.partition(sample, kth)[kth]
    return float(np.std(out, ddof=1))


def monte_carlo_var(
    book: Book,
    market: Market,
    cov: pd.DataFrame,
    alpha: float | None = None,
    horizon_days: float = 1.0,
    n_scenarios: int = 50_000,
    dist: str = "normal",
    df: float = 6.0,
    jumps: JumpSpec | None = None,
    seed: int | np.random.Generator = 0,
    option_method: str = "full",
    *,
    n_boot: int = 200,
    repair: str | None = None,
    confidence: float | None = None,
) -> MonteCarloVaRResult:
    """Monte Carlo VaR/ES with full revaluation of the book.

    One call simulates once.  VaR, ES and the bootstrap SE all come from
    the same P&L vector.  ``cov`` must cover every factor in
    ``book.factors(market)`` (extra columns are ignored).  ``alpha`` is
    the tail probability (default 0.01).

    Raises
    ------
    ValueError
        Empty book, missing factors, bad inputs, or a non-finite revalued
        P&L.
    """
    alpha = resolve_alpha(alpha, confidence)
    validate_horizon(horizon_days)
    require_positions(book)
    factors = book.factors(market)
    if not factors:
        raise ValueError("book has no risk factors (base-currency cash only)")
    missing = [f for f in factors if f not in cov.columns]
    if missing:
        raise ValueError(f"cov is missing required factor columns: {missing}")
    rng = _as_rng(seed)
    sub = cov.loc[factors, factors]
    scen = simulate_factor_returns(sub, n_scenarios, dist, df, jumps, rng,
                                   horizon_days, repair=repair)
    pnl = np.asarray(book.pnl(market, scen, option_method=option_method), dtype=float)
    pnl = _check_pnl(pnl)
    var, es = empirical_var_es(pnl, alpha)
    se = var_standard_error_bootstrap(pnl, alpha, n_boot=n_boot, seed=rng) \
        if pnl.size >= 10 else float("nan")
    return MonteCarloVaRResult(var, es, alpha, horizon_days, dist, int(n_scenarios), se, pnl)
