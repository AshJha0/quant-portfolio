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

Parameters follow the Basel III final framework (CRE31, CRE32): corporate
PD floor 0.05% (``pd_floor``), effective maturity floored at 1 year and
capped at 5 years, the 1.25 asset-correlation multiplier for large or
unregulated financial institutions (``financial_multiplier``), no 1.06
scaling factor, and K = max(0, LGD - EL_best) for defaulted exposures.
Pass ``pd_floor=0.0003`` for the Basel II floor.  Note K(PD) is NOT monotone over [0, 1]: it peaks around
PD ~ 30-40% and falls as PD -> 1 because expected loss (already provisioned)
absorbs an ever larger share of the 99.9% quantile loss.

Vasicek one-factor model
------------------------
Asset value ``A_i = sqrt(rho) * Z + sqrt(1-rho) * eps_i``; default iff
``A_i < N^{-1}(PD)``.  For an infinitely granular homogeneous portfolio the
default-rate CDF is::

    F(x) = N( ( sqrt(1-rho) * N^{-1}(x) - N^{-1}(PD) ) / sqrt(rho) )

with quantile ``x_q = N( (N^{-1}(PD) + sqrt(rho) * N^{-1}(q)) / sqrt(1-rho) )``
(the Basel K conditional-PD term is exactly ``x_0.999``). A finite
portfolio adds idiosyncratic (granularity) risk. ``granularity_adjustment``
gives its O(1/N) effect on the quantile (Gordy 2003; Gordy and
Lutkebohmert 2013) and ``finite_portfolio_quantile`` gives the exact
quantile of a homogeneous finite portfolio by quadrature. For realistic
portfolio sizes the adjustment is small next to Monte Carlo noise at the
99.9% level; use ``mc_quantile_ci`` to report that noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

__all__ = [
    "asset_correlation",
    "basel_k",
    "basel_report",
    "economic_capital",
    "el_by_bucket",
    "expected_loss",
    "finite_portfolio_quantile",
    "granularity_adjustment",
    "maturity_adjustment_b",
    "mc_quantile_ci",
    "risk_weighted_assets",
    "simulate_portfolio_losses",
    "vasicek_cdf",
    "vasicek_quantile",
]


def _validate_pd_lgd_ead(
    pd_: np.ndarray | float, lgd: np.ndarray | float, ead: np.ndarray | float
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
    pd_: np.ndarray | float,
    sales_millions: np.ndarray | float | None = None,
    financial_multiplier: np.ndarray | bool = False,
) -> np.ndarray:
    """Basel corporate asset correlation R(PD) with optional SME size adjustment.

    ``sales_millions``: annual sales S in EUR millions; the adjustment
    ``-0.04 * (1 - (S-5)/45)`` applies with S clamped to [5, 50] (None = large
    corporate, no adjustment). ``financial_multiplier`` (scalar or per-exposure
    booleans) applies the 1.25 multiplier of CRE31.5 to exposures to large
    regulated (total assets >= USD 100bn) or unregulated financial
    institutions.
    """
    p = np.asarray(pd_, dtype=float)
    if (~np.isfinite(p)).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("PD must lie in [0, 1]")
    w = (1.0 - np.exp(-50.0 * p)) / (1.0 - np.exp(-50.0))
    r = 0.12 * w + 0.24 * (1.0 - w)
    if sales_millions is not None:
        s = np.clip(np.asarray(sales_millions, dtype=float), 5.0, 50.0)
        r = r - 0.04 * (1.0 - (s - 5.0) / 45.0)
    fm = np.asarray(financial_multiplier, dtype=bool)
    return np.where(fm, 1.25 * r, r)


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
    pd_floor: float = 0.0005,
    financial_multiplier: np.ndarray | bool = False,
    el_best: np.ndarray | float | None = None,
) -> np.ndarray:
    """Basel IRB capital requirement K (fraction of EAD) for corporate exposures.

    Parameters
    ----------
    pd_ : array-like
        1-year PD, floored at ``pd_floor``. PD = 1 marks a defaulted
        exposure, for which K = max(0, LGD - EL_best).
    lgd : array-like
        Downturn LGD (decimal).
    maturity : array-like
        Effective maturity M in years, floored at 1 and capped at 5.
    sales_millions : optional
        SME size adjustment input, see :func:`asset_correlation`.
    pd_floor : float
        Regulatory PD floor (Basel III corporate: 0.05%).
    financial_multiplier : bool or array of bool
        Apply the 1.25 correlation multiplier (see :func:`asset_correlation`).
    el_best : optional
        Best estimate of expected loss for defaulted exposures (default:
        equal to LGD, so K = 0 for a defaulted exposure).

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
    if (~np.isfinite(m)).any() or (m < 0).any():
        raise ValueError("maturity must be finite and non-negative")
    if not (np.isfinite(pd_floor) and 0.0 <= pd_floor < 1.0):
        raise ValueError("pd_floor must lie in [0, 1)")
    m = np.clip(m, 1.0, 5.0)
    defaulted = p >= 1.0
    p_live = np.clip(np.where(defaulted, 0.5, p), pd_floor, 1.0 - 1e-12)
    r = asset_correlation(p_live, sales_millions, financial_multiplier)
    b = maturity_adjustment_b(p_live)
    cond_pd = norm.cdf(
        (norm.ppf(p_live) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1.0 - r)
    )
    k = (l * cond_pd - p_live * l) * (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)
    k = np.maximum(k, 0.0)
    if defaulted.any():
        elb = l if el_best is None else np.asarray(el_best, dtype=float)
        if (~np.isfinite(elb)).any() or (np.asarray(elb) < 0).any():
            raise ValueError("el_best must be finite and non-negative")
        k_def = np.maximum(np.broadcast_to(l, k.shape) - np.broadcast_to(elb, k.shape), 0.0)
        k = np.where(defaulted, k_def, k)
    return k


def risk_weighted_assets(k: np.ndarray | float, ead: np.ndarray | float) -> np.ndarray:
    """RWA = K * 12.5 * EAD (Basel III: no 1.06 scaling factor)."""
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
            "R": asset_correlation(np.clip(p, 0.0005, 0.9999)),
            "b": maturity_adjustment_b(np.clip(p, 0.0005, 0.9999)),
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


def finite_portfolio_quantile(
    q: float, pd_: float, rho: float, n_loans: int, lgd: float = 1.0
) -> float:
    """Exact q-quantile of the loss rate of a homogeneous finite portfolio.

    With n equal loans, the number of defaults given the factor Z is
    Binomial(n, p(Z)) with p(Z) = N((N^{-1}(PD) - sqrt(rho) Z) / sqrt(1-rho)).
    The unconditional CDF ``P(D <= k) = E[BinomCDF(k; n, p(Z))]`` is
    integrated over Z on a 40,001-point grid on [-10, 10] (composite
    trapezoid; the integrand is smooth and the omitted tails carry less
    than 1e-22 of mass). The quantile is the smallest k/n with
    P(D <= k) >= q, times LGD. Use it as the reference for Monte Carlo and
    for :func:`granularity_adjustment`.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")
    if not (0.0 < pd_ < 1.0):
        raise ValueError("PD must lie in (0, 1)")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie in (0, 1)")
    if int(n_loans) != n_loans or n_loans < 1:
        raise ValueError("n_loans must be a positive integer")
    if not (np.isfinite(lgd) and 0.0 <= lgd <= 1.0):
        raise ValueError("LGD must lie in [0, 1]")
    n = int(n_loans)
    z = np.linspace(-10.0, 10.0, 40_001)
    wz = norm.pdf(z) * (z[1] - z[0])
    wz[[0, -1]] *= 0.5
    pz = norm.cdf((norm.ppf(pd_) - np.sqrt(rho) * z) / np.sqrt(1.0 - rho))

    def cdf(k: int) -> float:
        return float(binom.cdf(k, n, pz) @ wz)

    lo, hi = -1, n  # cdf(lo) < q <= cdf(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cdf(mid) >= q:
            hi = mid
        else:
            lo = mid
    return lgd * hi / n


def granularity_adjustment(
    pd_: np.ndarray | float,
    lgd: np.ndarray | float,
    ead: np.ndarray | float,
    rho: float,
    q: float = 0.999,
    lgd_var_xi: float = 0.0,
    n_loans: int | None = None,
) -> float:
    """Granularity adjustment to the q-quantile loss rate (Gordy 2003;
    Gordy and Lutkebohmert 2013) in the one-factor Vasicek model.

    With Y = -Z the systematic factor (loss increases in Y), conditional
    mean mu(y) and conditional variance s2(y) of the portfolio loss rate,
    the second-order expansion of the quantile is

        GA = -1 / (2 phi(y_q)) * d/dy [ phi(y) s2(y) / mu'(y) ]  at y_q = N^{-1}(q),

    with exposure shares w_i = EAD_i / sum EAD,
    p_i(y) = N((N^{-1}(PD_i) + sqrt(rho) y) / sqrt(1 - rho)),
    mu(y) = sum w_i LGD_i p_i(y) and
    s2(y) = sum w_i^2 [(LGD_i^2 + V_i) p_i(y) - LGD_i^2 p_i(y)^2],
    where V_i = xi LGD_i (1 - LGD_i) is the LGD variance (Gordy and
    Lutkebohmert use xi = 0.25; the default 0 means deterministic LGD).
    The derivative is taken by a central difference with step 1e-4
    (relative error about 1e-8). The finite-portfolio quantile is
    approximately ``LGD-weighted vasicek_quantile + GA``. GA is O(1/N).

    Returns the adjustment as a loss rate (fraction of total EAD).
    """
    if not (0.0 < rho < 1.0):
        raise ValueError("rho must lie in (0, 1)")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")
    if not (np.isfinite(lgd_var_xi) and lgd_var_xi >= 0):
        raise ValueError("lgd_var_xi must be finite and >= 0")
    p = np.atleast_1d(np.asarray(pd_, dtype=float))
    if n_loans is not None:
        if p.size != 1:
            raise ValueError("n_loans only valid with scalar pd/lgd/ead")
        p = np.full(n_loans, float(p[0]))
        lgd = np.full(n_loans, float(np.asarray(lgd, dtype=float)))
        ead = np.full(n_loans, float(np.asarray(ead, dtype=float)))
    p, l, e = _validate_pd_lgd_ead(p, lgd, ead)
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("PD must lie strictly in (0, 1)")
    if e.sum() <= 0:
        raise ValueError("total EAD is zero")
    w = e / e.sum()
    v = lgd_var_xi * l * (1.0 - l)
    a0 = norm.ppf(p)
    s_r, s_1r = np.sqrt(rho), np.sqrt(1.0 - rho)

    def g(y: float) -> float:
        arg = (a0 + s_r * y) / s_1r
        pi = norm.cdf(arg)
        dmu = float(np.sum(w * l * norm.pdf(arg)) * s_r / s_1r)
        s2 = float(np.sum(w**2 * ((l**2 + v) * pi - l**2 * pi**2)))
        return norm.pdf(y) * s2 / dmu

    yq = float(norm.ppf(q))
    h = 1e-4
    deriv = (g(yq + h) - g(yq - h)) / (2.0 * h)
    return float(-deriv / (2.0 * norm.pdf(yq)))


def mc_quantile_ci(
    losses: np.ndarray, q: float = 0.999, conf: float = 0.95
) -> tuple[float, float, float]:
    """Monte Carlo quantile with a distribution-free order-statistic CI.

    The number of simulated losses below the true quantile is
    Binomial(n, q), so the order statistics at ranks given by the
    (1 -/+ conf)/2 binomial quantiles bracket it with coverage >= conf.
    Returns (estimate, lower, upper).
    """
    arr = np.sort(np.asarray(losses, dtype=float).ravel())
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("losses must be a non-empty finite array")
    if not (0.0 < q < 1.0 and 0.0 < conf < 1.0):
        raise ValueError("q and conf must lie in (0, 1)")
    n = arr.size
    a = (1.0 - conf) / 2.0
    lo_rank = int(binom.ppf(a, n, q))  # 0-based index of lower order statistic
    hi_rank = int(binom.ppf(1.0 - a, n, q))
    lo_rank = min(max(lo_rank - 1, 0), n - 1)
    hi_rank = min(max(hi_rank, 0), n - 1)
    return float(np.quantile(arr, q)), float(arr[lo_rank]), float(arr[hi_rank])
