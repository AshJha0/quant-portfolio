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
  ``Cov(r_fx, r_u) = Cov(r_fx) x`` and ``h* = 1``: full hedging.
* Safe havens (JPY, CHF) are NEGATIVELY correlated with risk assets: their
  unhedged exposure already offsets equity risk, so ``h* < 1`` (underhedge),
  possibly ``h* < 0`` (buy extra safe-haven exposure).  Full hedging is NOT
  variance-optimal in general.

Forward carry (the rate differential earned or paid on the hedge) shifts
the MEAN of hedged returns, not the variance, so the variance-minimising
``h*`` does not depend on it. ``hedged_returns(..., forward_carry=...)``
adds it to the hedged return series.

``h*`` is an in-sample OLS estimate. ``hedge_ratio_inference`` gives its
standard errors and ``out_of_sample_hedge`` evaluates a split-sample
estimate on held-out data.
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
    if not np.all(np.isfinite(exposures.to_numpy(dtype=float))):
        raise ValueError("exposures contain NaN/Inf")
    if not np.all(np.isfinite(fx_returns.to_numpy(dtype=float))):
        raise ValueError(
            "fx_returns contain NaN/Inf; align and clean the panel first "
            "(a single missing FX day yields an all-NaN hedge ratio)"
        )
    if not np.all(np.isfinite(unhedged.to_numpy(dtype=float))):
        raise ValueError("unhedged returns contain NaN/Inf")


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
    forward_carry: pd.DataFrame | pd.Series | None = None,
) -> pd.Series:
    """Return series after hedging with forwards.

    Selling a unit of foreign currency forward earns the forward points as
    well as the negative of the spot move:

        r_hedged = r_u - sum_i h_i x_i (r_fx,i - c_i),

    where ``c_i`` is the per-period forward carry of the hedge, the rate
    differential ``r_base - r_foreign,i`` (covered interest parity). With
    ``forward_carry=None`` the carry term is omitted and the result is the
    spot-only hedge. Its variance is the same but its mean is wrong by the
    carry.

    ``hedge_ratios`` may be a scalar (uniform hedge, e.g. 0 = unhedged,
    1 = full hedge) or a per-currency Series. ``forward_carry`` may be a
    per-currency Series (constant per-period carry) or a DataFrame aligned
    with ``fx_returns``.
    """
    _validate(unhedged, fx_returns, exposures)
    if np.isscalar(hedge_ratios):
        h = pd.Series(float(hedge_ratios), index=exposures.index)
    else:
        h = pd.Series(hedge_ratios).reindex(exposures.index)
        if h.isna().any():
            raise ValueError("hedge_ratios missing some exposure currencies")
    notional = h * exposures
    out = unhedged - (fx_returns * notional).sum(axis=1)
    if forward_carry is not None:
        if isinstance(forward_carry, pd.DataFrame):
            c = forward_carry.reindex(index=fx_returns.index, columns=fx_returns.columns)
            if not np.all(np.isfinite(c.to_numpy(dtype=float))):
                raise ValueError("forward_carry missing or non-finite for some dates/currencies")
            out = out + (c * notional).sum(axis=1)
        else:
            cs = pd.Series(forward_carry).reindex(exposures.index)
            if not np.all(np.isfinite(cs.to_numpy(dtype=float))):
                raise ValueError("forward_carry missing or non-finite for some currencies")
            out = out + float((cs * notional).sum())
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
    h=1 are feasible points): tested.
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


@dataclass
class HedgeInference:
    """OLS inference on the variance-minimising hedge ratios.

    Attributes
    ----------
    hedge_ratios : pd.Series
        h*_i (same as :func:`optimal_hedge_ratios`).
    std_errors : pd.Series
        Classical OLS standard errors of h*_i (iid residuals).
    t_stats : pd.Series
        (h*_i - 1) / SE_i: the t-statistic against a full hedge.
    """

    hedge_ratios: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series


def hedge_ratio_inference(
    unhedged: pd.Series,
    fx_returns: pd.DataFrame,
    exposures: pd.Series,
) -> HedgeInference:
    """Standard errors for h* from the regression r_u = a + sum_i H_i r_fx,i + e.

    H* is the OLS slope vector, so Var(H*) = s^2 (X_c'X_c)^-1 with X_c the
    demeaned FX returns and s^2 the residual variance on T - K - 1 degrees
    of freedom. h*_i = H*_i / x_i, so SE(h*_i) = SE(H*_i) / |x_i|. The
    errors assume iid residuals. They understate the uncertainty when
    residuals are autocorrelated or heteroskedastic.
    """
    _validate(unhedged, fx_returns, exposures)
    fx = fx_returns.to_numpy(dtype=float)
    fxc = fx - fx.mean(axis=0)
    ru = unhedged.to_numpy(dtype=float)
    ruc = ru - ru.mean()
    t, k = fx.shape
    if t <= k + 1:
        raise ValueError(f"need more than {k + 1} observations, got {t}")
    xtx = fxc.T @ fxc
    try:
        h_not = np.linalg.solve(xtx, fxc.T @ ruc)
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("FX covariance singular (pegged/duplicated currency?)") from exc
    resid = ruc - fxc @ h_not
    s2 = float(resid @ resid) / (t - k - 1)
    se_not = np.sqrt(np.diag(xtx_inv) * s2)
    x = exposures.to_numpy(dtype=float)
    h = pd.Series(h_not / x, index=exposures.index, name="hedge_ratio")
    se = pd.Series(se_not / np.abs(x), index=exposures.index, name="std_error")
    return HedgeInference(h, se, ((h - 1.0) / se).rename("t_vs_full_hedge"))


@dataclass
class OutOfSampleHedge:
    """Split-sample evaluation of the hedging decision.

    Attributes
    ----------
    hedge_ratios_train : pd.Series
        h* estimated on the training block only.
    hedge_ratios_test : pd.Series
        h* re-estimated on the test block (to show stability).
    reduction_full, reduction_optimal : float
        Variance reduction vs unhedged on the TEST block, for h = 1 and for
        the training-block h*.
    """

    hedge_ratios_train: pd.Series
    hedge_ratios_test: pd.Series
    reduction_full: float
    reduction_optimal: float


def out_of_sample_hedge(
    unhedged: pd.Series,
    fx_returns: pd.DataFrame,
    exposures: pd.Series,
    train_frac: float = 0.5,
) -> OutOfSampleHedge:
    """Estimate h* on the first ``train_frac`` of the sample, evaluate on the rest.

    No information from the test block enters the training estimate, so
    ``reduction_optimal`` is an out-of-sample variance reduction.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    _validate(unhedged, fx_returns, exposures)
    cut = int(len(unhedged) * train_frac)
    if cut < fx_returns.shape[1] + 2 or len(unhedged) - cut < fx_returns.shape[1] + 2:
        raise ValueError("train or test block too short for the number of currencies")
    u_tr, u_te = unhedged.iloc[:cut], unhedged.iloc[cut:]
    f_tr, f_te = fx_returns.iloc[:cut], fx_returns.iloc[cut:]
    h_tr = optimal_hedge_ratios(u_tr, f_tr, exposures)
    h_te = optimal_hedge_ratios(u_te, f_te, exposures)
    v0 = float(u_te.var(ddof=1))
    v1 = float(hedged_returns(u_te, f_te, exposures, 1.0).var(ddof=1))
    vo = float(hedged_returns(u_te, f_te, exposures, h_tr).var(ddof=1))
    return OutOfSampleHedge(h_tr, h_te, 1.0 - v1 / v0, 1.0 - vo / v0)
