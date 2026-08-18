"""Optimal currency hedging for an international portfolio.

Setup
-----
A base-currency investor holds an international portfolio whose UNHEDGED
base-currency return is ``r_u,t`` and which carries exposure ``x_i`` (portfolio
weight) to each foreign currency *i* with base-currency FX return ``r_fx,i,t``
(CCY vs base, so positive = foreign currency appreciates).  Selling forward a
fraction ``h_i`` (the hedge ratio) of each currency exposure gives

    r_hedged(h) = r_u - sum_i h_i * x_i * r_fx,i .

Minimising ``Var(r_hedged)`` over the hedge notionals ``H_i = h_i x_i`` is an
OLS projection with the closed form

    H* = Cov(r_fx)^{-1} Cov(r_fx, r_u),        h*_i = H*_i / x_i .

Classic results reproduced and tested here:

* If local (currency-hedged) returns are uncorrelated with FX, then
  ``Cov(r_fx, r_u) = Cov(r_fx) x`` and ``h* = 1`` — full hedging.
* Safe havens (JPY, CHF) are NEGATIVELY correlated with risk assets: their
  unhedged exposure already offsets equity risk, so ``h* < 1`` (underhedge),
  possibly ``h* < 0`` (buy extra safe-haven exposure).  Full hedging is NOT
  variance-optimal in general.

Forward carry cost (the rate differential paid on the hedge) shifts the MEAN
of hedged returns, not the variance; the variance-minimising ``h*`` is
therefore carry-free, and carry enters the decision as a separate
mean adjustment (documented in METHODOLOGY.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _validate(
    unhedged: pd.Series, fx_returns: pd.DataFrame, exposures: pd.Series
) -> None:
    if list(fx_returns.columns) != list(exposures.index):
        raise ValueError("fx_returns columns and exposures index must match")
    if not unhedged.index.equals(fx_returns.index):
        raise ValueError("unhedged and fx_returns must share the same index")
    if np.any(exposures.to_numpy() == 0):
        raise ValueError(
            "zero currency exposure: hedge ratio undefined; drop that currency"
        )


def optimal_hedge_ratios(
    unhedged: pd.Series,
    fx_returns: pd.DataFrame,
    exposures: pd.Series,
) -> pd.Series:
    """Variance-minimising hedge ratio per currency (closed form).

    Parameters
    ----------
    unhedged : pd.Series
        Base-currency returns of the unhedged portfolio.
    fx_returns : pd.DataFrame
        FX log returns (CCY vs base) per exposure currency.
    exposures : pd.Series
        Portfolio weight held in each currency (non-zero), indexed like
        ``fx_returns`` columns.

    Returns
    -------
    pd.Series
        ``h*_i = [Cov(fx)^-1 Cov(fx, r_u)]_i / x_i``; ``h=1`` is a full
        hedge, ``h<1`` underhedged, ``h>1`` overhedged.
    """
    _validate(unhedged, fx_returns, exposures)
    fx = fx_returns.to_numpy()
    fxc = fx - fx.mean(axis=0)
    ru = unhedged.to_numpy() - unhedged.to_numpy().mean()
    t = len(ru)
    cov_fx = fxc.T @ fxc / (t - 1)
    cov_fu = fxc.T @ ru / (t - 1)
    try:
        h_notional = np.linalg.solve(cov_fx, cov_fu)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "FX covariance singular (pegged/duplicated currency?)"
        ) from exc
    return pd.Series(
        h_notional / exposures.to_numpy(), index=exposures.index, name="hedge_ratio"
    )


def hedged_returns(
    unhedged: pd.Series,
    fx_returns: pd.DataFrame,
    exposures: pd.Series,
    hedge_ratios: pd.Series | float,
) -> pd.Series:
    """Return series after hedging: ``r_u - sum_i h_i x_i r_fx,i``.

    ``hedge_ratios`` may be a scalar (uniform hedge, e.g. 0 = unhedged,
    1 = full hedge) or a per-currency Series.
    """
    _validate(unhedged, fx_returns, exposures)
    if np.isscalar(hedge_ratios):
        h = pd.Series(float(hedge_ratios), index=exposures.index)
    else:
        h = pd.Series(hedge_ratios).reindex(exposures.index)
        if h.isna().any():
            raise ValueError("hedge_ratios missing some exposure currencies")
    out = unhedged - (fx_returns * (h * exposures)).sum(axis=1)
    out.name = "hedged"
    return out


@dataclass
class HedgeReport:
    """Variance decomposition of the hedging decision (per-period variances).

    Attributes
    ----------
    hedge_ratios : pd.Series
        Optimal per-currency hedge ratios h*.
    var_unhedged, var_full, var_optimal : float
        Portfolio return variance with h=0, h=1 and h=h*.
    reduction_full, reduction_optimal : float
        Fractional variance reduction vs unhedged for the full and optimal
        hedges (``1 - var/var_unhedged``).
    """

    hedge_ratios: pd.Series
    var_unhedged: float
    var_full: float
    var_optimal: float
    reduction_full: float
    reduction_optimal: float


def variance_decomposition(
    unhedged: pd.Series,
    fx_returns: pd.DataFrame,
    exposures: pd.Series,
) -> HedgeReport:
    """Compare unhedged / fully hedged / optimally hedged variance.

    By construction ``var_optimal <= min(var_unhedged, var_full)`` up to
    solver precision (h* minimises variance over ALL hedge ratios, and h=0,
    h=1 are feasible points) — tested.
    """
    h_opt = optimal_hedge_ratios(unhedged, fx_returns, exposures)
    v0 = float(unhedged.var(ddof=1))
    v1 = float(hedged_returns(unhedged, fx_returns, exposures, 1.0).var(ddof=1))
    vo = float(hedged_returns(unhedged, fx_returns, exposures, h_opt).var(ddof=1))
    return HedgeReport(
        hedge_ratios=h_opt,
        var_unhedged=v0,
        var_full=v1,
        var_optimal=vo,
        reduction_full=1.0 - v1 / v0,
        reduction_optimal=1.0 - vo / v0,
    )
