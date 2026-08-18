"""Calibrate Heston to an FX smile grid (5 pillars x expiries).

The objective is vega-weighted implied-vol residuals: for each pillar
strike the COS model price is inverted to a Garman-Kohlhagen implied
vol and compared with the market pillar vol.  Vega weighting is the
desk standard - it makes the objective approximately price-RMSE while
keeping the residuals in interpretable vol points, and it naturally
downweights the illiquid 10-delta wings and the very short expiry.

Identifiability (the kappa/xi ridge)
------------------------------------
Vanilla smiles pin down rho and v0 tightly (skew sign/level and
short-dated ATM), and the *combination* xi^2/kappa (long-run smile
convexity) much more tightly than kappa and xi separately: doubling
kappa and scaling xi by ~sqrt(2) moves vanillas very little.  The
ground-truth recovery test therefore asserts tight tolerances on
(rho, v0), a moderate one on theta, and loose ones on kappa and xi
individually - this is a property of the instrument set, not of the
optimiser, and is documented in docs/VALIDATION.md.

Typical FX pattern (encoded in the synthetic presets): EURUSD-like
markets calibrate to small |rho| (mild, fairly symmetric smile);
USDJPY-like markets to large negative rho (persistent JPY-call skew).
Rho flips sign across pairs - e.g. USD-EM pairs calibrate to positive
rho (devaluation skew on the call side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .garman_kohlhagen import gk_vega, implied_vol
from .heston import HestonParams, price_cos

__all__ = ["CalibrationSlice", "CalibrationResult", "heston_smile_vols", "calibrate_heston"]


@dataclass(frozen=True)
class CalibrationSlice:
    """One expiry of calibration targets."""

    T: float
    r_d: float
    r_f: float
    strikes: np.ndarray  # (n,)
    vols: np.ndarray  # (n,) market implied vols, decimals

    def __post_init__(self) -> None:
        object.__setattr__(self, "strikes", np.asarray(self.strikes, dtype=float))
        object.__setattr__(self, "vols", np.asarray(self.vols, dtype=float))
        if self.strikes.shape != self.vols.shape:
            raise ValueError("strikes and vols must have the same shape")
        if self.T <= 0:
            raise ValueError(f"T must be positive, got {self.T}")


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_vol_pts: float
    max_err_vol_pts: float
    n_evaluations: int
    success: bool
    market_vols: np.ndarray
    model_vols: np.ndarray
    weights: np.ndarray

    def summary(self) -> str:
        p = self.params
        return (
            f"v0={p.v0:.5f} kappa={p.kappa:.3f} theta={p.theta:.5f} "
            f"xi={p.xi:.3f} rho={p.rho:+.3f} | rmse={self.rmse_vol_pts:.3f} vol pts "
            f"(max {self.max_err_vol_pts:.3f}) | Feller ratio {p.feller_ratio:.2f}"
        )


def heston_smile_vols(
    S: float, sl: CalibrationSlice, params: HestonParams, N: int = 256
) -> np.ndarray:
    """Model implied vols at one expiry's strikes (COS price -> GK vol).

    A model price outside the no-arbitrage band (possible for extreme
    trial parameters mid-optimisation) maps to the bracket edge rather
    than NaN, so the optimiser sees a smooth, finite residual.
    """
    prices = price_cos(S, sl.strikes, sl.T, sl.r_d, sl.r_f, params, cp=-1, N=N)
    out = np.empty(len(prices))
    for i, (K, pr) in enumerate(zip(sl.strikes, prices)):
        iv = implied_vol(pr, S, float(K), sl.T, sl.r_d, sl.r_f, cp=-1, on_fail="nan")
        if math.isnan(iv):
            iv = 1e-4 if pr < 1e-12 else 5.0
        out[i] = iv
    return out


_DEFAULT_BOUNDS = {
    "v0": (1e-4, 2.0),
    "kappa": (0.05, 25.0),
    "theta": (1e-4, 2.0),
    "xi": (0.01, 4.0),
    "rho": (-0.999, 0.999),
}


def _default_x0(S: float, slices: list[CalibrationSlice]) -> np.ndarray:
    """Market-implied starting point: v0/theta from short/long ATM vols,
    rho from the average skew sign."""
    short, long_ = slices[0], slices[-1]
    v0 = float(np.median(short.vols)) ** 2
    theta = float(np.median(long_.vols)) ** 2
    skews = [float(sl.vols[-1] - sl.vols[0]) for sl in slices]
    rho0 = float(np.clip(8.0 * np.mean(skews), -0.9, 0.9))
    return np.array([v0, 2.0, theta, 0.5, rho0])


def calibrate_heston(
    S: float,
    slices: list[CalibrationSlice],
    x0: np.ndarray | None = None,
    vega_weighted: bool = True,
    bounds: dict | None = None,
    N: int = 256,
    max_nfev: int = 400,
) -> CalibrationResult:
    """Least-squares Heston calibration to a pillar-vol grid.

    Parameters
    ----------
    S : float
        Spot.
    slices : list of CalibrationSlice
        The (strike, vol) targets per expiry (typically 5 pillars x 6
        expiries from the broker quotes).
    x0 : ndarray, optional
        (v0, kappa, theta, xi, rho) start; defaults to a market-implied
        heuristic.
    vega_weighted : bool
        Weight residuals by GK vega at the market vol (normalised to
        max 1, floored at 0.05 so the wings still matter).
    N : int
        COS terms per pricing call.
    max_nfev : int
        Objective-evaluation budget for the optimiser.

    Returns
    -------
    CalibrationResult
    """
    if len(slices) == 0:
        raise ValueError("need at least one calibration slice")
    slices = sorted(slices, key=lambda s: s.T)
    b = dict(_DEFAULT_BOUNDS)
    if bounds:
        b.update(bounds)
    lb = np.array([b[k][0] for k in ("v0", "kappa", "theta", "xi", "rho")])
    ub = np.array([b[k][1] for k in ("v0", "kappa", "theta", "xi", "rho")])
    if x0 is None:
        x0 = _default_x0(S, slices)
    x0 = np.clip(np.asarray(x0, dtype=float), lb * 1.0000001, ub * 0.9999999)

    mkt = np.concatenate([sl.vols for sl in slices])
    if vega_weighted:
        w = np.concatenate(
            [
                np.array(
                    [
                        gk_vega(S, float(K), sl.T, sl.r_d, sl.r_f, float(v))
                        for K, v in zip(sl.strikes, sl.vols)
                    ]
                )
                for sl in slices
            ]
        )
        w = np.maximum(w / np.max(w), 0.05)
    else:
        w = np.ones_like(mkt)

    n_eval = 0

    def residuals(x: np.ndarray) -> np.ndarray:
        nonlocal n_eval
        n_eval += 1
        params = HestonParams(v0=x[0], kappa=x[1], theta=x[2], xi=x[3], rho=x[4])
        model = np.concatenate(
            [heston_smile_vols(S, sl, params, N=N) for sl in slices]
        )
        return w * (model - mkt)

    res = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        x_scale=np.array([0.01, 1.0, 0.01, 0.1, 0.1]),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    params = HestonParams(v0=res.x[0], kappa=res.x[1], theta=res.x[2],
                          xi=res.x[3], rho=res.x[4])
    model = np.concatenate([heston_smile_vols(S, sl, params, N=N) for sl in slices])
    err = (model - mkt) * 100.0  # vol points
    return CalibrationResult(
        params=params,
        rmse_vol_pts=float(np.sqrt(np.mean(err**2))),
        max_err_vol_pts=float(np.max(np.abs(err))),
        n_evaluations=n_eval,
        success=bool(res.success),
        market_vols=mkt,
        model_vols=model,
        weights=w,
    )
