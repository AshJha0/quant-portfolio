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
    "LEE_SLOPE_MAX",
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
    """Result of an SVI slice fit.

    ``lee_ok`` is True when the wing slopes satisfy Lee's bound
    ``b (1 + |rho|) <= 2``.  ``active_constraints`` names the constraints and
    box bounds binding at the optimum (for example ``"lee"``, ``"butterfly"``,
    ``"rho_bound"``).  An active constraint means the data asked for more
    than an arbitrage-free SVI can give.
    """

    params: SVIParams
    rmse_vol: float  # RMSE in implied-vol points (absolute, e.g. 0.001 = 0.1 vp)
    rmse_w: float  # RMSE in total-variance units
    arb_free: bool  # Durrleman g >= 0 on the check grid
    min_g: float
    n_points: int
    T: float
    restarts_tried: int = 0
    best_cost: float = np.inf
    all_costs: list = field(default_factory=list)
    lee_ok: bool = True
    active_constraints: tuple = ()
    constrained: bool = True


#: Lee (2004) bound on the total-variance wing slope ``b (1 + |rho|)``.
LEE_SLOPE_MAX = 2.0

#: Offsets (in units of SVI sigma) around the vertex m where Durrleman's g is
#: also evaluated: g changes on that scale, so a fixed k grid can miss it.
_VERTEX_GRID = np.linspace(-6.0, 6.0, 241)
_VERTEX_COARSE = np.linspace(-6.0, 6.0, 25)


def _svi_residuals(x: np.ndarray, k: np.ndarray, w: np.ndarray, wts: np.ndarray) -> np.ndarray:
    a, b, rho, m, sig = x
    d = k - m
    model = a + b * (rho * d + np.sqrt(d * d + sig * sig))
    return (model - w) * wts


def _zeliade_inner(k, w, wts, m, sig):
    """Quasi-explicit step (Zeliade 2009).

    For fixed ``(m, sigma)`` raw SVI is linear in ``(a, d, c)`` with
    ``y = (k - m)/sigma`` and ``w = a + d y + c sqrt(y^2 + 1)``, where
    ``c = b sigma`` and ``d = rho b sigma``.  Returns the weighted
    least-squares raw parameters, clipped into the Lee / raw-SVI domain,
    and their cost.
    """
    y = (k - m) / sig
    A = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)]) * wts[:, None]
    sol, *_ = np.linalg.lstsq(A, w * wts, rcond=None)
    a, d, c = (float(v) for v in sol)
    c = max(c, 1e-10)
    d = float(np.clip(d, -0.999 * c, 0.999 * c))
    shrink = min(1.0, LEE_SLOPE_MAX * sig / (c + abs(d)))  # Lee: c + |d| <= 2 sigma
    c, d = c * shrink, d * shrink
    b, rho = c / sig, d / c
    resid = (a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig * sig)) - w) * wts
    return np.array([a, b, rho, m, sig]), float(resid @ resid)


def _g_raw(x, kk):
    a, b, rho, m, sig = x
    d = kk - m
    sq = np.sqrt(d * d + sig * sig)
    ww = a + b * (rho * d + sq)
    wp = b * (rho + d / sq)
    wpp = b * sig * sig / sq**3
    return (1.0 - kk * wp / (2.0 * ww)) ** 2 - wp * wp / 4.0 * (1.0 / ww + 0.25) + wpp / 2.0


def _durrleman_min(x, k_lo: float, k_hi: float) -> tuple[float, np.ndarray]:
    """Minimum of Durrleman's g on [k_lo, k_hi] for raw SVI ``x``.

    A dense grid (plus a grid scaled by sigma around the vertex) locates the
    local minima, and each is polished with a bounded scalar minimisation,
    so a dip between grid points is not missed.  Returns the minimum and
    the polished local-minimum locations.
    """
    from scipy.optimize import minimize_scalar

    kk = np.unique(np.concatenate([np.linspace(k_lo, k_hi, 1601),
                                   np.clip(x[3] + x[4] * _VERTEX_GRID, k_lo, k_hi)]))
    with np.errstate(all="ignore"):
        g = _g_raw(x, kk)
    if not np.all(np.isfinite(g)):
        return -np.inf, kk[~np.isfinite(g)]
    idx = [i for i in range(len(kk))
           if (i == 0 or g[i] <= g[i - 1]) and (i == len(kk) - 1 or g[i] <= g[i + 1])]
    locs, best = [], float(np.min(g))
    for i in sorted(idx, key=lambda j: g[j])[:8]:
        lo, hi = kk[max(i - 1, 0)], kk[min(i + 1, len(kk) - 1)]
        if hi > lo:
            r = minimize_scalar(lambda t: float(_g_raw(x, np.array([t]))[0]),
                                bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
            locs.append(float(r.x))
            best = min(best, float(r.fun))
        else:
            locs.append(float(kk[i]))
    return best, np.array(locs)


def fit_svi(
    k: np.ndarray,
    total_variance: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
    n_restarts: int = 8,
    seed: int = 0,
    arbitrage_free: bool = True,
    check_margin: float = 1.0,
) -> SVIFitResult:
    """Fit raw SVI to one expiry.

    With ``arbitrage_free=True`` (default) the fit is constrained:

    * raw-SVI domain: ``b >= 0``, ``|rho| < 1``, ``sigma > 0`` and minimum
      total variance ``a + b sigma sqrt(1 - rho^2) > 0``;
    * Lee's wing bound ``b (1 + |rho|) <= 2`` in total-variance units.  In
      variance units this reads ``b_var (1 + |rho|) <= 2/T``, inside the
      Rogers-Tehranchi necessary bound ``4/T``;
    * no butterfly arbitrage: Durrleman's ``g(k) >= 0`` (the condition in
      Gatheral and Jacquier 2014, Thm 2.1) on a grid covering the quotes
      plus ``check_margin`` in k on each side.  Under the Lee bound the wing
      limit of g is ``1/4 - slope^2/16 >= 0``, so the far wings are covered.

    Starting points come from the Zeliade quasi-explicit split: on a grid
    of ``(m, sigma)`` the linear ``(a, d, c)`` problem is solved in closed
    form, and the best ``n_restarts`` grid points seed an SLSQP solve of
    the full constrained problem.

    With ``arbitrage_free=False`` the fit is bounded least squares with
    random restarts and a post-hoc Durrleman warning (the old behaviour).

    Parameters
    ----------
    k : array_like
        Forward log-moneyness of the quotes.
    total_variance : array_like
        Market total implied variance ``sigma_imp^2 * T`` at each ``k``.
        Non-finite entries are dropped.
    T : float
        Time to expiry in years.
    weights : array_like, optional
        Per-point weights on the total-variance residuals.  Default
        ``1 / (2 sqrt(w T))``, the first-order map from a total-variance
        error to an implied-vol error, so residuals are in vol units.  Pass
        vega or inverse bid-ask weights to change the emphasis.
    n_restarts : int
        Number of starting points.
    seed : int
        Seed for the unconstrained path's restart perturbations.
    arbitrage_free : bool
        Impose the constraints above (default) or fit unconstrained.
    check_margin : float
        Extra log-moneyness beyond the quote range on which the butterfly
        constraint and the reported check are evaluated.

    Raises
    ------
    ValueError
        Fewer than 5 valid quotes (raw SVI has 5 parameters) or invalid
        inputs.
    RuntimeError
        No start produced a feasible fit.
    """
    from scipy.optimize import minimize

    if not (np.isfinite(T) and T > 0.0):
        raise ValueError(f"T must be positive and finite, got {T}")
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if k.shape != w.shape:
        raise ValueError("k and total_variance must have the same shape")
    mask = np.isfinite(k) & np.isfinite(w)
    k, w = k[mask], w[mask]
    if k.size < 5:
        raise ValueError(
            f"SVI has 5 parameters; need at least 5 valid quotes per expiry, got {k.size}. "
            "Widen the strike range or use a simpler slice model."
        )
    if np.any(w <= 0.0):
        raise ValueError("total variance must be positive at every quote")
    if weights is None:
        wts = 1.0 / (2.0 * np.sqrt(w * T))
    else:
        wts = np.asarray(weights, dtype=float)[mask]
        if np.any(~np.isfinite(wts)) or np.any(wts < 0.0):
            raise ValueError("weights must be finite and non-negative")

    k_lo, k_hi = float(k.min()) - check_margin, float(k.max()) + check_margin
    span = max(float(k.max() - k.min()), 0.1)
    lb = np.array([-1.0, 1e-8, -0.999, k.min() - span, 1e-4])
    ub = np.array([max(2.0 * w.max(), 1.0), 10.0, 0.999, k.max() + span, 5.0])

    if not arbitrage_free:
        return _fit_svi_unconstrained(k, w, wts, T, lb, ub, n_restarts, seed, k_lo, k_hi)

    cands = []
    for m in np.linspace(k.min(), k.max(), 15):
        for sig in np.geomspace(0.01, max(span, 0.05), 12):
            x, c = _zeliade_inner(k, w, wts, m, sig)
            if x[0] + x[1] * x[4] * np.sqrt(1.0 - x[2] ** 2) > 0.0:
                cands.append((c, x))
    cands.sort(key=lambda t: t[0])
    starts = [x for _, x in cands[: max(n_restarts, 1)]]
    # A flat smile is always feasible; include it so SLSQP has at least one
    # start inside the constraint set.
    starts.append(np.array([float(np.mean(w)), 1e-6, 0.0, float(np.mean(k)), 0.1]))

    # Butterfly constraint points: a coarse k grid, the quotes, and a grid
    # around the vertex m scaled by sigma (g changes on that scale).  After
    # each solve, g is checked on a fine grid and any negative local minima
    # are added as constraint points (cutting-plane refinement).
    k_base = np.union1d(np.linspace(k_lo, k_hi, 61), k)
    k_fine = np.linspace(k_lo, k_hi, 1601)
    scale = 1.0 / max(float(np.mean(wts * wts)) * k.size * float(np.mean(w)) ** 2, 1e-300)

    def obj(x):
        r = _svi_residuals(x, k, w, wts)
        return float(r @ r) * scale

    def obj_grad(x):
        a, b, rho, m, sig = x
        d = k - m
        sq = np.sqrt(d * d + sig * sig)
        r = _svi_residuals(x, k, w, wts) * wts
        J = np.column_stack(
            [np.ones_like(k), rho * d + sq, b * d, -b * (rho + d / sq), b * sig / sq]
        )
        return 2.0 * scale * (J.T @ r)

    def g_at(x, kk):
        a, b, rho, m, sig = x
        d = kk - m
        sq = np.sqrt(d * d + sig * sig)
        ww = a + b * (rho * d + sq)
        wp = b * (rho + d / sq)
        wpp = b * sig * sig / sq**3
        return (1.0 - kk * wp / (2.0 * ww)) ** 2 - wp * wp / 4.0 * (1.0 / ww + 0.25) + wpp / 2.0

    def fine_grid(x):
        return np.concatenate([k_fine, x[3] + x[4] * _VERTEX_GRID])

    def solve(x0, extra, extra_rel):
        kc = np.concatenate([k_base, extra])
        rel = np.concatenate([_VERTEX_COARSE, extra_rel])
        cons = [
            {"type": "ineq", "fun": lambda x: LEE_SLOPE_MAX - x[1] * (1.0 + x[2])},
            {"type": "ineq", "fun": lambda x: LEE_SLOPE_MAX - x[1] * (1.0 - x[2])},
            {"type": "ineq",
             "fun": lambda x: x[0] + x[1] * x[4] * np.sqrt(max(1.0 - x[2] ** 2, 0.0)) - 1e-10},
            {"type": "ineq",
             # 1e-8 margin: SLSQP meets constraints to ~1e-9, the margin
             # keeps the returned slice at g >= 0 rather than -1e-9.
             "fun": lambda x: g_at(x, np.concatenate([kc, x[3] + x[4] * rel])) - 1e-8},
        ]
        with np.errstate(all="ignore"):
            return minimize(obj, x0, jac=obj_grad, method="SLSQP",
                            bounds=list(zip(lb, ub)), constraints=cons,
                            options={"maxiter": 300, "ftol": 1e-14}).x

    def feasible(x):
        with np.errstate(all="ignore"):
            return bool(
                np.all(np.isfinite(x))
                and x[1] * (1.0 + abs(x[2])) <= LEE_SLOPE_MAX + 1e-9
                and x[0] + x[1] * x[4] * np.sqrt(max(1.0 - x[2] ** 2, 0.0)) > 0.0
                and _durrleman_min(x, k_lo, k_hi)[0] >= -1e-10
            )

    best, best_val, costs, tried = None, np.inf, [], 0
    for x0 in starts:
        x0 = np.clip(x0, lb + 1e-10, ub - 1e-10)
        extra, extra_rel = np.empty(0), np.empty(0)
        x = x0
        try:
            for _ in range(12):
                x = solve(x, extra, extra_rel)
                if not np.all(np.isfinite(x)):
                    break
                gmin, locs = _durrleman_min(x, k_lo, k_hi)
                if gmin >= 0.0:
                    break
                # Dips near the vertex move with (m, sigma): keep each new
                # point both in absolute k and in vertex units (k - m)/sigma.
                extra = np.union1d(extra, locs)
                extra_rel = np.union1d(extra_rel, (locs - x[3]) / x[4])
        except (ValueError, ArithmeticError):
            continue
        tried += 1
        val = obj(x)
        costs.append(val)
        if val < best_val and feasible(x):
            best, best_val = x, val
    if best is None:
        raise RuntimeError("constrained SVI fit found no feasible solution; check input quotes")

    active = []
    a, b, rho, m, sig = best
    if b * (1.0 + abs(rho)) >= LEE_SLOPE_MAX - 1e-6:
        active.append("lee")
    if _durrleman_min(best, k_lo, k_hi)[0] <= 1e-8:
        active.append("butterfly")
    for name, xi, lo_, hi_ in zip(("a", "b", "rho", "m", "sigma"), best, lb, ub):
        if xi <= lo_ + 1e-8 * max(1.0, abs(lo_)) or xi >= hi_ - 1e-8 * max(1.0, abs(hi_)):
            active.append(f"{name}_bound")
    return _make_result(best, k, w, T, tried, best_val, costs, k_lo, k_hi, True, tuple(active))


def _make_result(x, k, w, T, tried, cost, costs, k_lo, k_hi, constrained, active):
    a, b, rho, m, sig = x
    params = SVIParams(float(a), float(b), float(rho), float(m), float(sig))
    w_fit = np.asarray(svi_total_variance(k, params))
    rmse_w = float(np.sqrt(np.mean((w_fit - w) ** 2)))
    rmse_vol = float(np.sqrt(np.mean((np.sqrt(w_fit / T) - np.sqrt(w / T)) ** 2)))
    min_g, _ = _durrleman_min(np.asarray(x, dtype=float), k_lo, k_hi)
    arb_free = min_g >= -1e-10  # constraint tolerance; g is O(1) near ATM
    lee_ok = bool(b * (1.0 + abs(rho)) <= LEE_SLOPE_MAX + 1e-9)
    if not arb_free or not lee_ok:
        warnings.warn(
            f"fitted SVI slice (T={T:.4g}) is not arbitrage-free: min Durrleman g = "
            f"{min_g:.3e}, wing slope b(1+|rho|) = {b * (1 + abs(rho)):.3f}",
            UserWarning,
        )
    return SVIFitResult(
        params=params, rmse_vol=rmse_vol, rmse_w=rmse_w, arb_free=arb_free,
        min_g=min_g, n_points=int(k.size), T=float(T), restarts_tried=tried,
        best_cost=float(cost), all_costs=list(costs), lee_ok=lee_ok,
        active_constraints=active, constrained=constrained,
    )


def _fit_svi_unconstrained(k, w, wts, T, lb, ub, n_restarts, seed, k_lo, k_hi):
    """Bounded least squares with random restarts, no arbitrage constraints."""
    rng = np.random.default_rng(seed)
    w_min = float(w.min())
    k_at_min = float(k[np.argmin(w)])
    span = max(float(k.max() - k.min()), 0.1)
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
                _svi_residuals, x0, bounds=(lb, ub), args=(k, w, wts), method="trf",
                xtol=1e-14, ftol=1e-14, gtol=1e-14,
            )
        except (ValueError, ArithmeticError):
            continue
        tried += 1
        costs.append(float(res.cost))
        a, b, rho, m, sig = res.x
        if a + b * sig * np.sqrt(max(1.0 - rho * rho, 0.0)) <= 0.0:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        raise RuntimeError("SVI fit failed from every restart; check input quotes")
    return _make_result(best.x, k, w, T, tried, best.cost, costs, k_lo, k_hi, False, ())


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
