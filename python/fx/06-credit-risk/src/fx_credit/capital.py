"""Expected loss and capital for sovereign exposures: EL, Basel standardized
risk weights, and Vasicek (ASRF) economic capital at 99.9%.

Regulatory context (documented, and why internal models differ)
---------------------------------------------------------------
Under the Basel standardized approach, sovereign exposures rated AAA to AA-
carry a **0% risk weight**, and national discretion allows 0% for domestic-
currency sovereign debt regardless of rating — a political legacy (sovereigns
were assumed risk-free in their own currency).  Russia 1998 (domestic GKO
default) and Greece 2012 show that assumption failing.  Internal/economic
capital models therefore assign sovereigns a *positive* PD from an internal
rating (block 1's scorecard) and a **higher asset correlation** than
corporates: sovereign defaults are driven by common global factors (commodity
cycles, USD funding conditions, contagion), so tail losses are more systemic.
We use rho_sov = 0.30 vs the Basel corporate formula's 0.12-0.24 range and
unit-test that the higher correlation fattens the capital tail.

Vasicek / ASRF
--------------
Conditional PD given systematic factor draw x (adverse = large positive x):

.. math:: PD(x) = \\Phi\\!\\left(\\frac{\\Phi^{-1}(PD) + \\sqrt{\\rho}\\,x}
    {\\sqrt{1-\\rho}}\\right)

Economic capital per unit EAD at confidence alpha:
``K = LGD * [PD(Phi^{-1}(alpha)) - PD]`` (unexpected loss over expected loss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "SOVEREIGN_RHO",
    "STANDARDIZED_SOVEREIGN_RW",
    "expected_loss",
    "basel_corporate_correlation",
    "vasicek_conditional_pd",
    "vasicek_capital",
    "standardized_rw",
    "capital_table",
]

#: Asset correlation for sovereign exposures in the economic-capital model.
#: Above the Basel corporate cap (0.24): sovereign defaults cluster on global
#: factors (USD cycle, commodity busts, contagion) far more than corporate
#: defaults do.
SOVEREIGN_RHO: float = 0.30

#: Basel standardized risk weights for rated sovereigns (external ratings).
#: AAA..AA- -> 0%; A -> 20%; BBB -> 50%; BB/B -> 100%; CCC and below -> 150%.
STANDARDIZED_SOVEREIGN_RW: dict[str, float] = {
    "AAA": 0.00,
    "AA": 0.00,
    "A": 0.20,
    "BBB": 0.50,
    "BB": 1.00,
    "B": 1.00,
    "CCC": 1.50,
    "C": 1.50,
}


def expected_loss(
    pd_1y: np.ndarray | float,
    lgd: np.ndarray | float,
    ead: np.ndarray | float,
) -> np.ndarray | float:
    """Expected loss ``EL = PD * LGD * EAD`` (elementwise).

    Raises ``ValueError`` on PD or LGD outside [0, 1] or negative EAD.
    """
    p = np.asarray(pd_1y, dtype=float)
    l = np.asarray(lgd, dtype=float)
    e = np.asarray(ead, dtype=float)
    if not np.all((p >= 0) & (p <= 1)) or not np.all((l >= 0) & (l <= 1)):
        raise ValueError("PD and LGD must be in [0,1] (NaN/Inf rejected)")
    if not np.all((e >= 0) & np.isfinite(e)):
        raise ValueError("EAD must be finite and >= 0 (NaN/Inf rejected)")
    out = p * l * e
    return float(out) if out.ndim == 0 else out


def basel_corporate_correlation(pd_1y: np.ndarray | float) -> np.ndarray | float:
    """Basel IRB corporate asset correlation: interpolates 0.24 -> 0.12 in PD.

    ``rho = 0.12 w + 0.24 (1 - w)``, ``w = (1 - e^{-50 PD}) / (1 - e^{-50})``.
    """
    p = np.asarray(pd_1y, dtype=float)
    w = (1.0 - np.exp(-50.0 * p)) / (1.0 - np.exp(-50.0))
    rho = 0.12 * w + 0.24 * (1.0 - w)
    return float(rho) if rho.ndim == 0 else rho


def vasicek_conditional_pd(
    pd_1y: np.ndarray | float,
    rho: float,
    x: float,
) -> np.ndarray | float:
    """PD conditional on systematic factor draw ``x`` (adverse = positive).

    Identities (unit-tested): x=0 with pd via median factor; rho=0 returns
    the unconditional PD; monotone increasing in x and (for x>0, pd<1/2)
    in rho.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0,1)")
    if not np.isfinite(x):
        raise ValueError("systematic factor x must be finite")
    p = np.asarray(pd_1y, dtype=float)
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("pd must be in [0,1] (NaN/Inf rejected)")
    with np.errstate(divide="ignore"):
        thresh = norm.ppf(p)
    out = norm.cdf((thresh + np.sqrt(rho) * x) / np.sqrt(1.0 - rho))
    out = np.where(p == 0.0, 0.0, np.where(p == 1.0, 1.0, out))
    return float(out) if out.ndim == 0 else out


def vasicek_capital(
    pd_1y: np.ndarray | float,
    lgd: np.ndarray | float,
    rho: float,
    alpha: float = 0.999,
) -> np.ndarray | float:
    """Economic capital per unit EAD: ``K = LGD*(PD(x_alpha) - PD)``.

    ``x_alpha = Phi^{-1}(alpha)``; at alpha=0.999 this is the Basel-style
    99.9% one-year solvency standard.  K(pd=0) = 0 and K(pd=1) = 0 (a sure
    default is all expected loss, no unexpected loss) — both unit-tested.
    """
    if not 0.5 <= alpha < 1.0:
        raise ValueError("alpha must be in [0.5, 1)")
    p = np.asarray(pd_1y, dtype=float)
    l = np.asarray(lgd, dtype=float)
    if not np.all((l >= 0) & (l <= 1)):
        raise ValueError("lgd must be in [0,1] (NaN/Inf rejected)")
    x_a = norm.ppf(alpha)
    cond = vasicek_conditional_pd(p, rho, x_a)
    k = l * (np.asarray(cond) - p)
    k = np.maximum(k, 0.0)
    return float(k) if k.ndim == 0 else k


def standardized_rw(rating: str) -> float:
    """Basel standardized sovereign risk weight for a letter rating."""
    try:
        return STANDARDIZED_SOVEREIGN_RW[rating]
    except KeyError as exc:
        raise ValueError(
            f"unknown rating {rating!r}; known: {sorted(STANDARDIZED_SOVEREIGN_RW)}"
        ) from exc


def capital_table(
    ratings: list[str],
    pds: list[float],
    lgd: float = 0.45,
    ead: float = 100.0,
    alpha: float = 0.999,
) -> pd.DataFrame:
    """Per-rating capital comparison: EL, standardized RW capital (8% of RWA),
    Vasicek K with sovereign rho vs Basel corporate rho.

    Returns a DataFrame with one row per rating band, all per ``ead`` of
    exposure.  Shows the wedge between the regulatory 0% floor for high-grade
    sovereigns and a positive internal economic-capital number.
    """
    rows = []
    for r, p in zip(ratings, pds):
        rw = standardized_rw(r)
        rho_c = basel_corporate_correlation(p)
        rows.append(
            {
                "rating": r,
                "pd_1y": p,
                "el": expected_loss(p, lgd, ead),
                "std_rw": rw,
                "std_capital": 0.08 * rw * ead,
                "k_sovereign": vasicek_capital(p, lgd, SOVEREIGN_RHO, alpha) * ead,
                "k_corp_rho": vasicek_capital(p, lgd, rho_c, alpha) * ead,
            }
        )
    return pd.DataFrame(rows)
