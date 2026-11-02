"""Loss given default and exposure at default.

Workout LGD
-----------
The economic loss on a defaulted facility is the exposure minus the present
value of what is recovered, net of workout costs, discounted back to the
default date:

    LGD = 1 - sum_k (R_k - C_k) / (1 + d)^(t_k) / EAD

with recoveries R_k, direct costs C_k at times t_k (years after default) and
discount rate d. Reference: BCBS (2005), "Guidance on paragraph 468 of the
Framework Document" (downturn LGD); EBA/GL/2017/16 section 6.3.1 (realised
LGD, discounting at an interest rate plus a 5% add-on).

EAD and the credit conversion factor
------------------------------------
For a revolving facility with limit L and drawn amount D at the reference
date, EAD = D + CCF * (L - D). The realised CCF of a defaulted facility is

    CCF = (EAD_default - D_ref) / (L_ref - D_ref),

defined only when there is an undrawn amount at the reference date.
Reference: Basel II paragraph 316; EBA/GL/2017/16 section 6.3.2 (fixed
horizon approach, 12-month reference date).

Beta LGD fit
------------
LGD lives on [0, 1] and is often bimodal. The method-of-moments beta fit
matches mean m and variance v: a = m * k, b = (1 - m) * k with
k = m (1 - m) / v - 1 (needs v < m (1 - m)).
"""

from __future__ import annotations

import numpy as np


def workout_lgd(
    ead: float,
    recoveries: np.ndarray,
    costs: np.ndarray,
    times: np.ndarray,
    discount_rate: float,
) -> float:
    """Realised workout LGD of one defaulted facility (see module docstring).

    The result is not floored: a recovery above EAD (penalty interest, for
    example) gives a negative LGD, which EBA/GL/2017/16 para 155 says to
    floor at zero only in the model, not in the realised data.
    """
    if ead <= 0.0:
        raise ValueError(f"ead must be positive, got {ead}")
    r = np.asarray(recoveries, dtype=float)
    c = np.asarray(costs, dtype=float)
    t = np.asarray(times, dtype=float)
    if not (r.shape == c.shape == t.shape):
        raise ValueError("recoveries, costs and times must have the same shape")
    if np.any(t < 0.0):
        raise ValueError("cash-flow times must be on or after default")
    if discount_rate <= -1.0:
        raise ValueError("discount_rate must exceed -1")
    pv = float(np.sum((r - c) / (1.0 + discount_rate) ** t))
    return 1.0 - pv / ead


def ead_from_ccf(drawn: np.ndarray | float, limit: np.ndarray | float, ccf: np.ndarray | float) -> np.ndarray:
    """EAD = drawn + CCF * max(limit - drawn, 0)."""
    d = np.asarray(drawn, dtype=float)
    lim = np.asarray(limit, dtype=float)
    k = np.asarray(ccf, dtype=float)
    if np.any(k < 0.0) or np.any(k > 1.0):
        raise ValueError("ccf must lie in [0, 1]")
    return d + k * np.maximum(lim - d, 0.0)


def realised_ccf(drawn_ref: np.ndarray, limit_ref: np.ndarray, ead_default: np.ndarray) -> np.ndarray:
    """Realised CCF per facility; NaN where nothing was undrawn at the reference date."""
    d = np.asarray(drawn_ref, dtype=float)
    lim = np.asarray(limit_ref, dtype=float)
    e = np.asarray(ead_default, dtype=float)
    undrawn = lim - d
    out = np.full(np.broadcast(d, lim, e).shape, np.nan)
    ok = undrawn > 0.0
    out[ok] = ((e - d) / np.where(ok, undrawn, 1.0))[ok]
    return out


def beta_moments_fit(lgd: np.ndarray) -> tuple[float, float]:
    """Method-of-moments beta parameters (a, b) for LGD observations in (0, 1)."""
    x = np.asarray(lgd, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two observations")
    if np.any(x < 0.0) or np.any(x > 1.0):
        raise ValueError("LGD observations must lie in [0, 1]")
    m = float(np.mean(x))
    v = float(np.var(x, ddof=1))
    if v <= 0.0 or v >= m * (1.0 - m):
        raise ValueError("variance must lie in (0, m(1-m)) for a beta fit")
    k = m * (1.0 - m) / v - 1.0
    return m * k, (1.0 - m) * k


def downturn_lgd(lgd_by_year: np.ndarray, default_rate_by_year: np.ndarray, n_worst: int = 1) -> float:
    """Downturn LGD: the mean LGD over the ``n_worst`` years with the highest default rate.

    A simple observed-impact reading of EBA/RTS/2018/04 article 3 (downturn
    period identified by an economic factor, here the default rate itself).
    """
    lgd = np.asarray(lgd_by_year, dtype=float)
    dr = np.asarray(default_rate_by_year, dtype=float)
    if lgd.shape != dr.shape or lgd.ndim != 1:
        raise ValueError("lgd_by_year and default_rate_by_year must be 1-D and equal length")
    if not 1 <= n_worst <= lgd.size:
        raise ValueError("n_worst must be between 1 and the number of years")
    idx = np.argsort(-dr, kind="stable")[:n_worst]
    return float(np.mean(lgd[idx]))
