"""Per-expiry smile fitting: raw SVI with no-arbitrage diagnostics.

Raw SVI (Gatheral 2004) parameterises *total implied variance*
``w(k) = sigma_imp(k)^2 * T`` as a function of forward log-moneyness
``k = ln(K / F)``:

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

with ``b >= 0``, ``|rho| < 1``, ``sigma > 0`` and ``a + b*sigma*sqrt(1-rho^2) >= 0``
(so ``w > 0`` everywhere).

Butterfly (strike) arbitrage is diagnosed with the Durrleman condition: the
risk-neutral density is non-negative iff

    g(k) = (1 - k*w'/(2w))^2 - (w'^2/4) * (1/w + 1/4) + w''/2  >=  0

for all k, where primes are derivatives of total variance w.r.t. k.  For raw
SVI these derivatives are analytic:

    w'(k)  = b * ( rho + (k - m) / sqrt((k - m)^2 + sigma^2) )
    w''(k) = b * sigma^2 / ((k - m)^2 + sigma^2)^(3/2)

A quadratic-in-delta fit is provided as a naive baseline: it has no
no-arbitrage machinery and typically underfits the wings, which is exactly
why SVI is preferred (see docs/METHODOLOGY.md).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

__all__ = [
    "SVIParams",
    "SVIFitResult",
    "svi_total_variance",
    "svi_dw_dk",
    "svi_d2w_dk2",
    "svi_implied_vol",
    "durrleman_g",
    "check_butterfly",
    "fit_svi",
    "fit_quadratic_delta",
    "QuadraticDeltaFit",
]


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters for one expiry.

    Attributes
    ----------
    a : float
        Overall variance level (total-variance units).
    b : float
        Slope of the wings (>= 0).
    rho : float
        Skew / asymmetry, in (-1, 1).
    m : float
        Horizontal translation of the smile minimum (log-moneyness units).
    sigma : float
        Smoothness of the vertex (> 0).
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        if self.b < 0.0:
            raise ValueError(f"SVI b must be >= 0, got {self.b}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"SVI rho must be in (-1, 1), got {self.rho}")
        if self.sigma <= 0.0:
            raise ValueError(f"SVI sigma must be > 0, got {self.sigma}")
        min_w = self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)
        if min_w <= 0.0:
            raise ValueError(
                "SVI parameters give non-positive total variance: "
                f"a + b*sigma*sqrt(1-rho^2) = {min_w:.6g} <= 0"
            )

    def as_array(self) -> np.ndarray:
        """Return parameters as ``[a, b, rho, m, sigma]``."""
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])


def svi_total_variance(k: np.ndarray | float, p: SVIParams) -> np.ndarray | float:
    """Raw SVI total implied variance ``w(k) = sigma^2(k) * T``.

    Parameters
    ----------
    k : array_like
        Forward log-moneyness ``ln(K/F)``.
    p : SVIParams
        Raw SVI parameters.
    """
    k = np.asarray(k, dtype=float)
    d = k - p.m
    w = p.a + p.b * (p.rho * d + np.sqrt(d * d + p.sigma**2))
    return w if w.ndim else float(w)


def svi_dw_dk(k: np.ndarray | float, p: SVIParams) -> np.ndarray | float:
    """Analytic first derivative of SVI total variance w.r.t. log-moneyness."""
    k = np.asarray(k, dtype=float)
    d = k - p.m
    out = p.b * (p.rho + d / np.sqrt(d * d + p.sigma**2))
    return out if out.ndim else float(out)


def svi_d2w_dk2(k: np.ndarray | float, p: SVIParams) -> np.ndarray | float:
    """Analytic second derivative of SVI total variance w.r.t. log-moneyness."""
    k = np.asarray(k, dtype=float)
    d = k - p.m
    out = p.b * p.sigma**2 / (d * d + p.sigma**2) ** 1.5
    return out if out.ndim else float(out)


def svi_implied_vol(k: np.ndarray | float, p: SVIParams, T: float) -> np.ndarray | float:
    """Implied volatility from an SVI slice: ``sqrt(w(k) / T)``."""
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    return np.sqrt(svi_total_variance(k, p) / T)


def durrleman_g(k: np.ndarray | float, p: SVIParams) -> np.ndarray | float:
    """Durrleman butterfly-arbitrage function g(k) for an SVI slice.

    The slice is free of butterfly arbitrage (risk-neutral density >= 0) iff
    ``g(k) >= 0`` for all ``k``.  Uses the analytic SVI derivatives.
    """
    k = np.asarray(k, dtype=float)
    w = svi_total_variance(k, p)
    wp = svi_dw_dk(k, p)
    wpp = svi_d2w_dk2(k, p)
    g = (1.0 - k * wp / (2.0 * w)) ** 2 - (wp * wp / 4.0) * (1.0 / w + 0.25) + wpp / 2.0
    return g if g.ndim else float(g)


def check_butterfly(
    p: SVIParams,
    k_min: float = -1.5,
    k_max: float = 1.5,
    n_grid: int = 401,
) -> tuple[bool, float, np.ndarray]:
    """Check an SVI slice for butterfly arbitrage on a log-moneyness grid.

    Parameters
    ----------
    p : SVIParams
        Slice parameters.
    k_min, k_max : float
        Grid limits in log-moneyness.
    n_grid : int
        Number of grid points.

    Returns
    -------
    (is_arb_free, min_g, k_violations)
        ``is_arb_free`` is True when ``min_k g(k) >= 0`` on the grid;
        ``min_g`` is the grid minimum of g; ``k_violations`` are the grid
        points where ``g < 0`` (empty when arbitrage-free).
    """
    k = np.linspace(k_min, k_max, n_grid)
    g = np.asarray(durrleman_g(k, p))
    violations = k[g < 0.0]
    return bool(g.min() >= 0.0), float(g.min()), violations


@dataclass
class SVIFitResult:
    """Result of an SVI slice fit."""

    params: SVIParams
    rmse_vol: float  # RMSE in implied-vol points (absolute, e.g. 0.001 = 0.1 vp)
    rmse_w: float  # RMSE in total-variance units
    arb_free: bool  # Durrleman g >= 0 on the default grid
    min_g: float
    n_points: int
    T: float
    restarts_tried: int = 0
    best_cost: float = np.inf
    all_costs: list = field(default_factory=list)


def _svi_residuals(x: np.ndarray, k: np.ndarray, w: np.ndarray, wts: np.ndarray) -> np.ndarray:
    a, b, rho, m, sig = x
    d = k - m
    model = a + b * (rho * d + np.sqrt(d * d + sig * sig))
    return (model - w) * wts


def fit_svi(
    k: np.ndarray,
    total_variance: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
    n_restarts: int = 8,
    seed: int = 0,
) -> SVIFitResult:
    """Fit raw SVI to one expiry by constrained least squares with restarts.

    Parameters
    ----------
    k : array_like
        Forward log-moneyness of the quotes.
    total_variance : array_like
        Market total implied variance ``sigma_imp^2 * T`` at each ``k``.
        Non-finite entries are dropped.
    T : float
        Time to expiry in years (used only for vol-space RMSE reporting).
    weights : array_like, optional
        Per-point weights on the total-variance residuals (default: equal).
    n_restarts : int
        Number of random restarts around data-driven initial guesses.
    seed : int
        Seed for the restart perturbations (deterministic fits).

    Returns
    -------
    SVIFitResult
        Best-of-restarts parameters plus fit and arbitrage diagnostics.

    Raises
    ------
    ValueError
        If fewer than 5 valid quotes are supplied (raw SVI has 5 parameters).
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.shape != w.shape:
        raise ValueError("k and total_variance must have the same shape")
    mask = np.isfinite(k) & np.isfinite(w)
    k, w = k[mask], w[mask]
    if weights is None:
        wts = np.ones_like(k)
    else:
        wts = np.asarray(weights, dtype=float)[mask]
    if k.size < 5:
        raise ValueError(
            f"SVI has 5 parameters; need at least 5 valid quotes per expiry, got {k.size}. "
            "Widen the strike range or use a simpler slice model."
        )
    if np.any(w <= 0.0):
        raise ValueError("total variance must be positive at every quote")

    rng = np.random.default_rng(seed)
    w_min = float(w.min())
    k_at_min = float(k[np.argmin(w)])
    span = max(float(k.max() - k.min()), 0.1)

    # Bounds: a can be slightly negative (raw SVI allows it as long as w > 0),
    # b bounded above by 4/(T*(1+|rho|)) heuristics are skipped in favour of a
    # generous box; positivity of w is enforced post-hoc through SVIParams.
    lb = np.array([-1.0, 1e-8, -0.999, k.min() - span, 1e-4])
    ub = np.array([max(2.0 * w.max(), 1.0), 10.0, 0.999, k.max() + span, 5.0])

    base_guesses = [
        np.array([0.8 * w_min, 0.1, -0.5, k_at_min, 0.1]),
        np.array([0.5 * w_min, 0.3, -0.7, 0.0, 0.2]),
        np.array([0.9 * w_min, 0.05, 0.0, k_at_min, 0.05]),
    ]

    best = None
    costs: list[float] = []
    tried = 0
    for i in range(max(n_restarts, 1)):
        if i < len(base_guesses):
            x0 = base_guesses[i].copy()
        else:
            g = base_guesses[i % len(base_guesses)]
            x0 = g * (1.0 + 0.3 * rng.standard_normal(5))
            x0[2] = np.clip(g[2] + 0.4 * rng.standard_normal(), -0.9, 0.9)
            x0[3] = g[3] + 0.2 * rng.standard_normal() * span
        x0 = np.clip(x0, lb + 1e-10, ub - 1e-10)
        try:
            res = least_squares(
                _svi_residuals, x0, bounds=(lb, ub), args=(k, w, wts), method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14
            )
        except Exception:  # numerical failure of one start is not fatal
            continue
        tried += 1
        costs.append(float(res.cost))
        # Reject fits whose minimum total variance is non-positive.
        a, b, rho, m, sig = res.x
        if a + b * sig * np.sqrt(max(1.0 - rho * rho, 0.0)) <= 0.0:
            continue
        if best is None or res.cost < best.cost:
            best = res

    if best is None:
        raise RuntimeError("SVI fit failed from every restart; check input quotes")

    a, b, rho, m, sig = best.x
    params = SVIParams(float(a), float(b), float(rho), float(m), float(sig))
    w_fit = np.asarray(svi_total_variance(k, params))
    rmse_w = float(np.sqrt(np.mean((w_fit - w) ** 2)))
    vol_fit = np.sqrt(w_fit / T)
    vol_mkt = np.sqrt(w / T)
    rmse_vol = float(np.sqrt(np.mean((vol_fit - vol_mkt) ** 2)))
    k_lo, k_hi = float(k.min()) - 0.5, float(k.max()) + 0.5
    arb_free, min_g, _ = check_butterfly(params, k_lo, k_hi)
    if not arb_free:
        warnings.warn(
            f"fitted SVI slice (T={T:.4g}) violates the Durrleman butterfly condition: "
            f"min g = {min_g:.3e} < 0",
            UserWarning,
        )
    return SVIFitResult(
        params=params,
        rmse_vol=rmse_vol,
        rmse_w=rmse_w,
        arb_free=arb_free,
        min_g=min_g,
        n_points=int(k.size),
        T=float(T),
        restarts_tried=tried,
        best_cost=float(best.cost),
        all_costs=costs,
    )


@dataclass(frozen=True)
class QuadraticDeltaFit:
    """Naive baseline: implied vol quadratic in BS call delta.

    ``sigma(delta) = c0 + c1 * delta + c2 * delta^2`` where ``delta`` is the
    forward call delta ``N(d1)`` computed at each quote's own implied vol.
    No no-arbitrage structure; documented as a baseline only.
    """

    c0: float
    c1: float
    c2: float
    rmse_vol: float

    def __call__(self, delta: np.ndarray | float) -> np.ndarray | float:
        delta = np.asarray(delta, dtype=float)
        out = self.c0 + self.c1 * delta + self.c2 * delta * delta
        return out if out.ndim else float(out)


def fit_quadratic_delta(
    k: np.ndarray, vols: np.ndarray, T: float
) -> QuadraticDeltaFit:
    """Fit the naive quadratic-in-delta smile baseline.

    Parameters
    ----------
    k : array_like
        Forward log-moneyness.
    vols : array_like
        Implied vols at each ``k``.
    T : float
        Time to expiry in years.

    Returns
    -------
    QuadraticDeltaFit
        Callable fit object with in-sample RMSE in vol points.
    """
    k = np.asarray(k, dtype=float)
    vols = np.asarray(vols, dtype=float)
    mask = np.isfinite(k) & np.isfinite(vols)
    k, vols = k[mask], vols[mask]
    if k.size < 3:
        raise ValueError(f"need at least 3 valid quotes for a quadratic fit, got {k.size}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    srt = vols * np.sqrt(T)
    d1 = (-k + 0.5 * srt * srt) / srt
    delta = norm.cdf(d1)
    coeffs = np.polyfit(delta, vols, 2)  # highest power first
    fit_vals = np.polyval(coeffs, delta)
    rmse = float(np.sqrt(np.mean((fit_vals - vols) ** 2)))
    return QuadraticDeltaFit(c0=float(coeffs[2]), c1=float(coeffs[1]), c2=float(coeffs[0]), rmse_vol=rmse)
