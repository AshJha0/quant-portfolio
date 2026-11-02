"""Multi-factor (time-series regression) risk model.

Each asset's return is regressed on K observed factor returns:

    r_it = a_i + sum_k B_ik f_kt + e_it,

and the model covariance is Sigma = B Sigma_f B' + D, with Sigma_f the
sample factor covariance and D the diagonal of residual variances
(ddof = K + 1). Reference: Grinold and Kahn (2000), "Active Portfolio
Management", 2nd ed., chapter 3, eq. (3.16); Connor (1995), "The three
types of factor models", Financial Analysts Journal 51(3).

Risk decomposition of a portfolio w with factor exposures x = B' w:

    variance        = x' Sigma_f x + w' D w
    factor k share  = x_k (Sigma_f x)_k          (Euler; sums to x' Sigma_f x)
    asset i share   = w_i (Sigma w)_i / sigma_p   (sums to sigma_p)

The Euler identities hold exactly because variance is homogeneous of
degree two (Menchero and Davis 2011, "Risk contribution is exposure times
volatility times correlation", J. Portfolio Management 37(2)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FactorModel:
    alpha: np.ndarray
    loadings: np.ndarray
    factor_cov: np.ndarray
    specific_var: np.ndarray

    @property
    def covariance(self) -> np.ndarray:
        return self.loadings @ self.factor_cov @ self.loadings.T + np.diag(self.specific_var)


def fit_factor_model(returns: np.ndarray, factors: np.ndarray) -> FactorModel:
    """OLS time-series factor model; returns (T x N), factors (T x K)."""
    r = np.asarray(returns, dtype=float)
    f = np.asarray(factors, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    if r.ndim != 2 or f.shape[0] != r.shape[0]:
        raise ValueError("returns (T x N) and factors (T x K) must share T")
    t, k = f.shape
    if t <= k + 1:
        raise ValueError("need more observations than factors + 1")
    x = np.column_stack([np.ones(t), f])
    coef, *_ = np.linalg.lstsq(x, r, rcond=None)
    resid = r - x @ coef
    spec = np.sum(resid**2, axis=0) / (t - k - 1)
    fcov = np.atleast_2d(np.cov(f, rowvar=False))
    return FactorModel(coef[0], coef[1:].T, fcov, spec)


@dataclass(frozen=True)
class RiskDecomposition:
    total_vol: float
    factor_var: float
    specific_var: float
    factor_contrib: np.ndarray
    asset_contrib: np.ndarray


def decompose_risk(model: FactorModel, weights: np.ndarray) -> RiskDecomposition:
    w = np.asarray(weights, dtype=float)
    if w.shape != (model.loadings.shape[0],):
        raise ValueError("weights must have one entry per asset")
    x = model.loadings.T @ w
    fx = model.factor_cov @ x
    fvar = float(x @ fx)
    svar = float(w @ (model.specific_var * w))
    sigma = model.covariance
    vol = float(np.sqrt(fvar + svar))
    asset = w * (sigma @ w) / vol if vol > 0.0 else np.zeros_like(w)
    return RiskDecomposition(vol, fvar, svar, x * fx, asset)
