"""Expected loss, Basel IRB capital and the Vasicek one-factor loss model.

Expected loss
-------------
``EL_i = PD_i * LGD_i * EAD_i`` per loan; portfolio EL is the sum.  An
optional downturn-LGD haircut scales LGD up multiplicatively (capped at 1).

Basel IRB (corporate exposures, foundation/advanced)
----------------------------------------------------
Exact regulatory formulas (BCBS, "An Explanatory Note on the Basel II IRB
Risk Weight Functions", July 2005; Basel III framework CRE31 keeps the same
functional form for corporates):

* Asset correlation::

      R(PD) = 0.12 * (1 - e^{-50 PD}) / (1 - e^{-50})
            + 0.24 * [1 - (1 - e^{-50 PD}) / (1 - e^{-50})]

  with optional SME size adjustment ``- 0.04 * (1 - (S - 5)/45)`` for annual
  sales S in EUR millions, S clamped to [5, 50].

* Maturity adjustment ``b(PD) = (0.11852 - 0.05478 * ln PD)^2``.

* Capital requirement (as a fraction of EAD)::

      K = [ LGD * N( (N^{-1}(PD) + sqrt(R) * N^{-1}(0.999)) / sqrt(1-R) )
            - PD * LGD ] * (1 + (M - 2.5) * b) / (1 - 1.5 * b)

* ``RWA = K * 12.5 * EAD`` (Basel III: no 1.06 scaling factor).

Regulatory PD floor for corporates: 0.03% (Basel II) — applied via
``pd_floor``.  Note K(PD) is NOT monotone over [0, 1]: it peaks around
PD ~ 30-40% and falls as PD -> 1 because expected loss (already provisioned)
absorbs an ever larger share of the 99.9% quantile loss.

Vasicek one-factor model
------------------------
Asset value ``A_i = sqrt(rho) * Z + sqrt(1-rho) * eps_i``; default iff
``A_i < N^{-1}(PD)``.  For an infinitely granular homogeneous portfolio the
default-rate CDF is::

    F(x) = N( ( sqrt(1-rho) * N^{-1}(x) - N^{-1}(PD) ) / sqrt(rho) )

with quantile ``x_q = N( (N^{-1}(PD) + sqrt(rho) * N^{-1}(q)) / sqrt(1-rho) )``
(the Basel K conditional-PD term is exactly ``x_0.999``).  Monte Carlo on a
finite portfolio adds granularity (idiosyncratic) risk on top of the
systematic tail; the finite-portfolio 99.9% loss quantile therefore sits at
or above the analytic infinitely-granular one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "expected_loss",
    "el_by_bucket",
    "asset_correlation",
    "maturity_adjustment_b",
    "basel_k",
    "risk_weighted_assets",
    "basel_report",
    "vasicek_cdf",
    "vasicek_quantile",
    "simulate_portfolio_losses",
    "economic_capital",
]


def _validate_pd_lgd_ead(
    pd_: np.ndarray, lgd: np.ndarray, ead: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.atleast_1d(np.asarray(pd_, dtype=float))
    l = np.atleast_1d(np.asarray(lgd, dtype=float))
    e = np.atleast_1d(np.asarray(ead, dtype=float))
    p, l, e = np.broadcast_arrays(p, l, e)
    if p.size == 0:
        raise ValueError("empty portfolio: no loans supplied")
    if not (np.isfinite(p).all() and np.isfinite(l).all() and np.isfinite(e).all()):
        raise ValueError("PD, LGD and EAD must be finite (no NaN/Inf)")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("PD must lie in [0, 1]")
    if ((l < 0) | (l > 1)).any():
        raise ValueError("LGD must lie in [0, 1]")
    if (e < 0).any():
        raise ValueError("EAD must be non-negative")
    return p.astype(float), l.astype(float), e.astype(float)


def expected_loss(
    pd_: np.ndarray | float,
    lgd: np.ndarray | float,
    ead: np.ndarray | float,
    downturn_lgd_haircut: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Per-loan and portfolio expected loss (currency units of EAD).

    Parameters
    ----------
    pd_, lgd, ead : array-like (broadcastable)
        1-year PD (decimal), loss given default (decimal fraction of EAD),
        exposure at default (currency).
    downturn_lgd_haircut : float
        Multiplicative LGD uplift h >= 0: ``LGD_dt = min(LGD * (1 + h), 1)``.

    Returns
    -------
    (el_per_loan, portfolio_el)
    """
    p, l, e = _validate_pd_lgd_ead(pd_, lgd, ead)
    if downturn_lgd_haircut < 0:
        raise ValueError("downturn_lgd_haircut must be >= 0")
    l_eff = np.minimum(l * (1.0 + downturn_lgd_haircut), 1.0)
    el = p * l_eff * e
    return el, float(el.sum())


def el_by_bucket(
    df: pd.DataFrame,
    bucket_col: str,
    pd_col: str = "pd",
    lgd_col: str = "lgd",
    ead_col: str = "ead",
) -> pd.DataFrame:
    """Aggregate EL, EAD and average PD by a bucket column (e.g. rating)."""
    if len(df) == 0:
        raise ValueError("empty portfolio: no loans supplied")
    el, _ = expected_loss(df[pd_col].to_numpy(), df[lgd_col].to_numpy(), df[ead_col].to_numpy())
    tmp = df[[bucket_col, ead_col]].copy()
    tmp["el"] = el
    tmp["_pd_w"] = df[pd_col].to_numpy() * df[ead_col].to_numpy()
    out = tmp.groupby(bucket_col, observed=True).agg(
        n=(ead_col, "size"), ead=(ead_col, "sum"), el=("el", "sum"), _pd_w=("_pd_w", "sum")
    )
    out["avg_pd_ead_weighted"] = out["_pd_w"] / out["ead"]
    out["el_rate"] = out["el"] / out["ead"]
    return out.drop(columns="_pd_w").reset_index()


def asset_correlation(
    pd_: np.ndarray | float, sales_millions: np.ndarray | float | None = None
) -> np.ndarray:
    """Basel corporate asset correlation R(PD) with optional SME size adjustment.

    ``sales_millions``: annual sales S in EUR millions; the adjustment
    ``-0.04 * (1 - (S-5)/45)`` applies with S clamped to [5, 50] (None = large
    corporate, no adjustment).
    """
    p = np.asarray(pd_, dtype=float)
    if (~np.isfinite(p)).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("PD must lie in [0, 1]")
    w = (1.0 - np.exp(-50.0 * p)) / (1.0 - np.exp(-50.0))
    r = 0.12 * w + 0.24 * (1.0 - w)
    if sales_millions is not None:
        s = np.clip(np.asarray(sales_millions, dtype=float), 5.0, 50.0)
        r = r - 0.04 * (1.0 - (s - 5.0) / 45.0)
    return r


def maturity_adjustment_b(pd_: np.ndarray | float) -> np.ndarray:
    """Basel smoothed maturity adjustment b(PD) = (0.11852 - 0.05478 ln PD)^2."""
    p = np.asarray(pd_, dtype=float)
    if (p <= 0).any() or (p > 1).any():
        raise ValueError("PD must lie in (0, 1] for the maturity adjustment")
    return (0.11852 - 0.05478 * np.log(p)) ** 2


def basel_k(
    pd_: np.ndarray | float,
    lgd: np.ndarray | float,
    maturity: np.ndarray | float = 2.5,
    sales_millions: np.ndarray | float | None = None,
    pd_floor: float = 0.0003,
) -> np.ndarray:
    """Basel IRB capital requirement K (fraction of EAD) for corporate exposures.

    Parameters
    ----------
    pd_ : array-like
        1-year PD; floored at ``pd_floor`` (regulatory floor, default 0.03%)
        and capped at 0.9999 for numerical stability.
    lgd : array-like
        Downturn LGD (decimal).
    maturity : array-like
        Effective maturity M in years (regulatory range [1, 5]).
    sales_millions : optional
        SME size adjustment input, see :func:`asset_correlation`.
    pd_floor : float
        Regulatory PD floor.

    Returns
    -------
    np.ndarray of K values.
    """
    p = np.atleast_1d(np.asarray(pd_, dtype=float))
    l = np.asarray(lgd, dtype=float)
    m = np.asarray(maturity, dtype=float)
    if (~np.isfinite(p)).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("PD must lie in [0, 1]")
    if (~np.isfinite(l)).any() or ((l < 0) | (l > 1)).any():
        raise ValueError("LGD must lie in [0, 1]")
    if ((m < 0)).any():
        raise ValueError("maturity must be non-negative")
    p = np.clip(p, pd_floor, 0.9999)
    r = asset_correlation(p, sales_millions)
    b = maturity_adjustment_b(p)
    cond_pd = norm.cdf(
        (norm.ppf(p) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1.0 - r)
    )
    k = (l * cond_pd - p * l) * (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)
    return np.maximum(k, 0.0)


def risk_weighted_assets(k: np.ndarray | float, ead: np.ndarray | float) -> np.ndarray:
    """RWA = K * 12.5 * EAD (Basel III — no 1.06 scaling factor)."""
    ka = np.asarray(k, dtype=float)
    ea = np.asarray(ead, dtype=float)
    if (ka < 0).any() or (ea < 0).any():
        raise ValueError("K and EAD must be non-negative")
    return ka * 12.5 * ea


def basel_report(
    pd_bands: np.ndarray, lgd: float = 0.45, maturity: float = 2.5, ead: float = 1.0
) -> pd.DataFrame:
    """K, risk weight and RWA per unit EAD for a grid of PD bands."""
    p = np.asarray(pd_bands, dtype=float)
    k = basel_k(p, lgd, maturity)
    return pd.DataFrame(
        {
            "pd": p,
            "R": asset_correlation(np.clip(p, 0.0003, 0.9999)),
            "b": maturity_adjustment_b(np.clip(p, 0.0003, 0.9999)),
            "K": k,
            "risk_weight": k * 12.5,
            "rwa": risk_weighted_assets(k, ead),
        }
    )


def vasicek_cdf(x: np.ndarray | float, pd_: float, rho: float) -> np.ndarray:
    """CDF of the limiting (infinitely granular) portfolio default rate.

    ``F(x) = N( (sqrt(1-rho) N^{-1}(x) - N^{-1}(PD)) / sqrt(rho) )``.
    """
    if not (0.0 < pd_ < 1.0):
        raise ValueError("PD must lie in (0, 1)")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie in (0, 1)")
    xa = np.asarray(x, dtype=float)
    xa = np.clip(xa, 1e-15, 1.0 - 1e-15)
    return norm.cdf(
        (np.sqrt(1.0 - rho) * norm.ppf(xa) - norm.ppf(pd_)) / np.sqrt(rho)
    )


def vasicek_quantile(q: np.ndarray | float, pd_: float, rho: float) -> np.ndarray:
    """Quantile of the limiting default rate:
    ``x_q = N( (N^{-1}(PD) + sqrt(rho) N^{-1}(q)) / sqrt(1-rho) )``."""
    if not (0.0 < pd_ < 1.0):
        raise ValueError("PD must lie in (0, 1)")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie in (0, 1)")
    qa = np.clip(np.asarray(q, dtype=float), 1e-15, 1.0 - 1e-15)
    return norm.cdf(
        (norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(qa)) / np.sqrt(1.0 - rho)
    )


def simulate_portfolio_losses(
    pd_: np.ndarray | float,
    lgd: np.ndarray | float,
    ead: np.ndarray | float,
    rho: float,
    n_sims: int = 20_000,
    seed: int = 0,
    n_loans: int | None = None,
) -> np.ndarray:
    """Monte Carlo one-factor portfolio loss rates (loss / total EAD).  Seeded.

    Scalars with ``n_loans`` build a homogeneous portfolio; arrays give a
    heterogeneous one.  For each scenario a single systematic factor Z is
    drawn; loan i defaults iff
    ``sqrt(rho) Z + sqrt(1-rho) eps_i < N^{-1}(PD_i)``.

    Returns
    -------
    np.ndarray, shape (n_sims,)
        Portfolio loss as a fraction of total EAD per scenario.
    """
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie in (0, 1)")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    p = np.atleast_1d(np.asarray(pd_, dtype=float))
    if n_loans is not None:
        if p.size != 1:
            raise ValueError("n_loans only valid with scalar pd/lgd/ead")
        p = np.full(n_loans, float(p[0]))
        lgd = np.full(n_loans, float(np.asarray(lgd, dtype=float)))
        ead = np.full(n_loans, float(np.asarray(ead, dtype=float)))
    p, l, e = _validate_pd_lgd_ead(p, lgd, ead)
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("PD must lie strictly in (0, 1) for simulation")
    n = p.size
    rng = np.random.default_rng(seed)
    thresh = norm.ppf(p)
    total_ead = e.sum()
    if total_ead <= 0:
        raise ValueError(
            "total EAD is zero: portfolio loss rate (loss / total EAD) undefined"
        )
    losses = np.empty(n_sims)
    lw = l * e / total_ead  # loss weight per loan
    sq_rho, sq_1mrho = np.sqrt(rho), np.sqrt(1.0 - rho)
    # Vectorise over scenarios in blocks to bound memory.
    block = max(1, int(5_000_000 / max(n, 1)))
    done = 0
    while done < n_sims:
        b = min(block, n_sims - done)
        z = rng.standard_normal((b, 1))
        eps = rng.standard_normal((b, n))
        defaults = (sq_rho * z + sq_1mrho * eps) < thresh
        losses[done : done + b] = defaults @ lw
        done += b
    return losses


def economic_capital(
    losses_or_quantile: np.ndarray | float, el_rate: float, q: float = 0.999
) -> float:
    """Economic capital = loss quantile - expected loss (as loss rates).

    Pass either a simulated loss-rate array (quantile taken at ``q``) or a
    precomputed analytic quantile.
    """
    if not (0.0 < q <= 1.0):
        raise ValueError("q must lie in (0, 1]")
    if np.isscalar(losses_or_quantile):
        vq = float(losses_or_quantile)  # type: ignore[arg-type]
    else:
        arr = np.asarray(losses_or_quantile, dtype=float)
        if arr.size == 0:
            raise ValueError("empty loss array")
        vq = float(np.quantile(arr, q))
    return vq - el_rate
