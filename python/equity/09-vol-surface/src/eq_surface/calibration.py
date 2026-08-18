"""Heston calibration to an implied-volatility surface.

Objective
---------
Weighted least squares **in implied-vol space** (not price space): residuals
are ``(model_iv - market_iv) * weight`` with vega-proportional weights.
Vol-space residuals with vega weights approximate price-space errors while
keeping every expiry on a comparable scale; pure price-space least squares
over-weights long-dated ITM options, pure unweighted vol space over-weights
worthless wings whose quotes carry no information.

Optimisation
------------
``scipy.optimize.least_squares`` (trust-region reflective, box bounds) from
several deterministic-seeded starting points; the best local optimum wins.
Multi-start matters because the Heston objective has a curved, nearly flat
valley in the ``(kappa, xi)`` plane (see below).

Identifiability: the kappa/xi ridge
-----------------------------------
Vanilla smiles constrain Heston parameters through a small number of smile
features (level, skew, curvature and their term decay).  ``rho`` and ``v0``
map almost directly onto short-dated skew and ATM level and are well
identified.  ``kappa`` and ``xi`` are *jointly* identified mainly through the
term-structure decay of skew: raising mean reversion while raising vol-of-vol
leaves vanilla smiles almost unchanged (both scale the effective skew decay),
producing a ridge in parameter space.  The Jacobian at the optimum is
therefore ill-conditioned -- the reported condition number is typically
1e3-1e5 -- and day-over-day recalibration can walk along the ridge even when
the surface barely moves.  This is a *feature of vanilla information content*,
not an optimiser bug; the desk consequence (parameter-stability monitoring,
model reserves for ridge-sensitive exotics) is discussed in docs/DESK_GUIDE.md.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .black_scholes import bs_vega, implied_vol_vector
from .heston import FellerWarning, HestonParams, feller_condition, heston_call_gl

__all__ = ["CalibrationResult", "calibrate_heston", "heston_model_ivs"]

# Box bounds for (v0, kappa, theta, rho, xi).
_LB = np.array([1e-4, 0.05, 1e-4, -0.999, 1e-3])
_UB = np.array([1.0, 12.0, 1.0, 0.999, 3.0])


@dataclass
class CalibrationResult:
    """Result of a Heston surface calibration.

    Attributes
    ----------
    params : HestonParams
        Best-fit parameters.
    rmse_vol : float
        Overall weighted-fit RMSE in absolute vol units (0.01 = 1 vol point);
        computed unweighted over all quotes for reporting comparability.
    rmse_vol_points : float
        ``rmse_vol * 100`` -- RMSE in vol points.
    rmse_by_expiry : dict
        Expiry (years) -> unweighted RMSE in vol points for that expiry.
    condition_number : float
        Condition number of the (weighted) Jacobian at the optimum -- a
        direct read on parameter identifiability (large = ridge).
    jac_singular_values : ndarray
        Singular values of the Jacobian (descending).
    n_starts, best_start : int
        Number of starts run and index of the winner.
    start_costs : list of float
        Final cost of each start (for multi-modality diagnostics).
    success : bool
        Optimiser convergence flag of the winning start.
    feller_ratio : float
        ``2 kappa theta / xi^2`` of the fitted parameters.
    n_quotes : int
        Number of implied-vol quotes fitted.
    """

    params: HestonParams
    rmse_vol: float
    rmse_vol_points: float
    rmse_by_expiry: dict
    condition_number: float
    jac_singular_values: np.ndarray
    n_starts: int
    best_start: int
    start_costs: list = field(default_factory=list)
    success: bool = True
    feller_ratio: float = np.inf
    n_quotes: int = 0

    def report(self) -> str:
        """Human-readable calibration report."""
        p = self.params
        lines = [
            "Heston calibration report",
            "-" * 60,
            f"  v0    = {p.v0:.6f}   kappa = {p.kappa:.4f}   theta = {p.theta:.6f}",
            f"  rho   = {p.rho:.4f}   xi    = {p.xi:.4f}",
            f"  Feller ratio 2*kappa*theta/xi^2 = {self.feller_ratio:.3f}"
            + ("  (VIOLATED)" if self.feller_ratio < 1.0 else ""),
            f"  overall RMSE = {self.rmse_vol_points:.4f} vol points over {self.n_quotes} quotes",
            f"  Jacobian condition number = {self.condition_number:.3e}",
            "  RMSE by expiry (vol points):",
        ]
        for T, rmse in sorted(self.rmse_by_expiry.items()):
            lines.append(f"    T = {T:7.4f}y : {rmse:.4f}")
        return "\n".join(lines)


def heston_model_ivs(
    S: float,
    r: float,
    q: float,
    expiries: np.ndarray,
    strikes_by_expiry: list[np.ndarray],
    p: HestonParams,
) -> list[np.ndarray]:
    """Model implied vols for a quote grid (Fourier price -> BS inversion).

    Returns one array of implied vols per expiry (``nan`` where the model
    price cannot be inverted, e.g. worthless deep wings).
    """
    out = []
    for T, Ks in zip(expiries, strikes_by_expiry):
        prices = np.asarray(heston_call_gl(S, np.asarray(Ks, dtype=float), float(T), r, q, p))
        out.append(implied_vol_vector(prices, S, np.asarray(Ks, dtype=float), float(T), r, q, "call"))
    return out


def _flatten_market(
    expiries, strikes_by_expiry, ivs_by_expiry
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    expiries = np.asarray(expiries, dtype=float)
    if expiries.ndim != 1 or expiries.size == 0:
        raise ValueError("expiries must be a non-empty 1-D array")
    if np.any(expiries <= 0.0):
        raise ValueError("all expiries must be positive")
    if len(strikes_by_expiry) != expiries.size or len(ivs_by_expiry) != expiries.size:
        raise ValueError("strikes_by_expiry and ivs_by_expiry must match expiries in length")
    Ks, IVs = [], []
    for i, T in enumerate(expiries):
        K = np.asarray(strikes_by_expiry[i], dtype=float)
        iv = np.asarray(ivs_by_expiry[i], dtype=float)
        if K.shape != iv.shape:
            raise ValueError(f"expiry {T}: strikes and ivs shapes differ")
        mask = np.isfinite(iv) & (iv > 0.0) & np.isfinite(K) & (K > 0.0)
        if mask.sum() == 0:
            raise ValueError(f"expiry {T}: no valid quotes")
        Ks.append(K[mask])
        IVs.append(iv[mask])
    return expiries, Ks, IVs


def calibrate_heston(
    S: float,
    r: float,
    q: float,
    expiries: np.ndarray,
    strikes_by_expiry: list[np.ndarray],
    ivs_by_expiry: list[np.ndarray],
    n_starts: int = 4,
    seed: int = 0,
    x0: HestonParams | None = None,
) -> CalibrationResult:
    """Calibrate Heston (v0, kappa, theta, rho, xi) to an implied-vol surface.

    Parameters
    ----------
    S, r, q : float
        Spot, risk-free rate, dividend yield (continuous, annualised).
    expiries : array_like
        Expiries in years, one per slice.
    strikes_by_expiry : list of arrays
        Strike grid per expiry.
    ivs_by_expiry : list of arrays
        Market implied vols per expiry (``nan`` entries are dropped).
    n_starts : int
        Number of multi-start local optimisations (>= 1).
    seed : int
        Seed for start-point perturbations (deterministic calibration).
    x0 : HestonParams, optional
        Extra user-supplied starting point (prepended to the start list).

    Returns
    -------
    CalibrationResult
        Best-fit parameters plus fit-quality and identifiability diagnostics.
    """
    if S <= 0.0:
        raise ValueError(f"spot must be positive, got {S}")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    expiries, Ks, IVs = _flatten_market(expiries, strikes_by_expiry, ivs_by_expiry)
    n_quotes = int(sum(iv.size for iv in IVs))
    if n_quotes < 5:
        raise ValueError(f"need at least 5 quotes to identify 5 parameters, got {n_quotes}")

    # Vega weights from *market* vols (fixed during optimisation), mean-1.
    wts = []
    for T, K, iv in zip(expiries, Ks, IVs):
        w = np.array([bs_vega(S, k, float(T), r, q, v) for k, v in zip(K, iv)])
        wts.append(w)
    w_all = np.concatenate(wts)
    w_all = w_all / max(w_all.mean(), 1e-12)
    iv_all = np.concatenate(IVs)

    def residuals(x: np.ndarray) -> np.ndarray:
        v0, kappa, theta, rho, xi = x
        try:
            p = HestonParams(v0, kappa, theta, rho, xi)
        except ValueError:
            return np.full(n_quotes, 10.0)
        model = []
        for T, K in zip(expiries, Ks):
            prices = np.asarray(heston_call_gl(S, K, float(T), r, q, p))
            model.append(implied_vol_vector(prices, S, K, float(T), r, q, "call"))
        model_all = np.concatenate(model)
        res = (model_all - iv_all) * w_all
        # Unrecoverable model IVs (worthless wings under trial params): a
        # finite penalty keeps the optimiser away without breaking it.
        return np.where(np.isfinite(res), res, 1.0)

    # --- starting points --------------------------------------------------
    rng = np.random.default_rng(seed)
    # ATM-anchored heuristic: v0 from the shortest expiry, theta from the longest.
    atm_short = IVs[0][np.argmin(np.abs(np.log(Ks[0] / (S * np.exp((r - q) * expiries[0])))))]
    atm_long = IVs[-1][np.argmin(np.abs(np.log(Ks[-1] / (S * np.exp((r - q) * expiries[-1])))))]
    base = np.array([atm_short**2, 1.5, atm_long**2, -0.6, 0.5])
    starts = []
    if x0 is not None:
        starts.append(x0.as_array())
    starts.append(base)
    alt = [
        np.array([atm_short**2, 4.0, atm_long**2, -0.3, 1.0]),
        np.array([0.5 * atm_short**2 + 0.5 * atm_long**2, 0.8, atm_long**2, -0.8, 0.3]),
    ]
    starts.extend(alt)
    while len(starts) < n_starts + (1 if x0 is not None else 0):
        pert = base * (1.0 + 0.4 * rng.standard_normal(5))
        pert[3] = np.clip(-0.6 + 0.4 * rng.standard_normal(), -0.95, 0.5)
        starts.append(pert)
    starts = [np.clip(s, _LB + 1e-8, _UB - 1e-8) for s in starts[: max(n_starts, len(starts) if x0 else n_starts)]]
    starts = starts[: n_starts + (1 if x0 is not None else 0)]

    best = None
    best_idx = -1
    costs: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FellerWarning)
        for i, s in enumerate(starts):
            try:
                res = least_squares(
                    residuals, s, bounds=(_LB, _UB), method="trf",
                    xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400,
                )
            except Exception:
                costs.append(np.inf)
                continue
            costs.append(float(res.cost))
            if best is None or res.cost < best.cost:
                best, best_idx = res, i

    if best is None:
        raise RuntimeError("Heston calibration failed from every starting point")

    v0, kappa, theta, rho, xi = best.x
    params = HestonParams(float(v0), float(kappa), float(theta), float(rho), float(xi))
    feller = feller_condition(params, warn=True)  # warn-not-raise by design

    # Identifiability diagnostics from the optimum's Jacobian.
    sv = np.linalg.svd(best.jac, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-300))

    # Reporting RMSEs: unweighted, in vol space.
    model_ivs = heston_model_ivs(S, r, q, expiries, Ks, params)
    rmse_by_expiry = {}
    sq_all = []
    for T, iv_m, iv_k in zip(expiries, IVs, model_ivs):
        d = iv_k - iv_m
        d = d[np.isfinite(d)]
        rmse_by_expiry[float(T)] = float(np.sqrt(np.mean(d**2)) * 100.0)
        sq_all.append(d**2)
    rmse = float(np.sqrt(np.mean(np.concatenate(sq_all))))

    return CalibrationResult(
        params=params,
        rmse_vol=rmse,
        rmse_vol_points=rmse * 100.0,
        rmse_by_expiry=rmse_by_expiry,
        condition_number=cond,
        jac_singular_values=sv,
        n_starts=len(starts),
        best_start=best_idx,
        start_costs=costs,
        success=bool(best.success),
        feller_ratio=feller,
        n_quotes=n_quotes,
    )
