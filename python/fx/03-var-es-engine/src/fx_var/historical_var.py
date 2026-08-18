"""Historical-simulation VaR: plain HS, age-weighted (BRW), and filtered HS.

All three variants revalue the *actual book* (full revaluation through
:meth:`fx_var.book.Book.pnl`, including forwards' rate legs and options'
Garman-Kohlhagen repricing) under historical factor-return scenarios:

* **plain**    - each of the last T days is an equally weighted scenario.
* **age**      - Boudoukh-Richardson-Whitelaw exponential age weights
                 ``w_t \\propto lambda^{age}``: recent days dominate, so the
                 VaR reacts faster after a regime change.
* **fhs**      - Filtered Historical Simulation (Barone-Adesi et al.):
                 returns are devolatilised by a per-factor EWMA sigma and
                 rescaled to *today's* sigma forecast, preserving the
                 empirical cross-sectional dependence (copula) while making
                 the scenario set conditionally heteroscedastic.  This is
                 the variant that survives volatility clustering backtests
                 (see docs/VALIDATION.md).

Multi-day horizons use square-root-of-time scaling of the 1-day figure -
documented limitation for carry-trade books with negative skew
(docs/VALIDATION.md, failure mode F4).

Peg blindness: HS sees only what is in the window.  A pegged currency
contributes ~zero scenarios, so the engine emits
:class:`fx_var.common.PegBlindnessWarning` and the desk must add the
peg-break stress add-on (fx_var.stress_testing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .book import Book, Market
from .common import validate_alpha, validate_horizon, validate_returns, warn_peg_factors
from .expected_shortfall import empirical_var_es

__all__ = ["HistoricalVaRResult", "historical_var", "ewma_volatility"]


@dataclass(frozen=True)
class HistoricalVaRResult:
    """Result of a historical-simulation VaR run.

    ``var``/``es`` are positive losses in the book's base currency at the
    requested horizon; ``pnl`` and ``weights`` are the 1-day scenario P&L
    vector and scenario weights actually used for the quantile.
    """

    var: float
    es: float
    alpha: float
    horizon_days: float
    method: str
    pnl: np.ndarray
    weights: np.ndarray
    flagged_peg_factors: tuple[str, ...] = ()


def ewma_volatility(returns: pd.DataFrame, lam: float = 0.94) -> tuple[pd.DataFrame, pd.Series]:
    """RiskMetrics EWMA volatility per factor.

    ``sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r2_{t-1}`` - i.e.
    ``sigma_t`` is the *forecast* for day t using data through t-1, seeded
    with the full-sample standard deviation.

    Returns
    -------
    (sigma, sigma_next)
        ``sigma``: DataFrame of per-day forecast vols aligned with
        ``returns``; ``sigma_next``: Series with the one-step-ahead
        forecast after the last observation.
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lambda must be in (0, 1), got {lam}")
    r = returns.to_numpy(dtype=float)
    n, k = r.shape
    sig2 = np.empty((n + 1, k))
    sig2[0] = np.maximum(r.var(axis=0, ddof=1), 1e-18)
    for i in range(n):
        sig2[i + 1] = lam * sig2[i] + (1.0 - lam) * r[i] ** 2
    sigma = pd.DataFrame(np.sqrt(sig2[:-1]), index=returns.index, columns=returns.columns)
    sigma_next = pd.Series(np.sqrt(sig2[-1]), index=returns.columns)
    return sigma, sigma_next


def historical_var(
    book: Book,
    market: Market,
    returns: pd.DataFrame,
    alpha: float = 0.99,
    horizon_days: float = 1.0,
    method: str = "plain",
    decay: float = 0.995,
    ewma_lambda: float = 0.94,
    min_obs: int = 60,
    option_method: str = "full",
    warn_pegs: bool = True,
) -> HistoricalVaRResult:
    """Historical-simulation VaR/ES for ``book`` at ``market``.

    Parameters
    ----------
    book, market : Book, Market
        The book to revalue and the reference snapshot.
    returns : pandas.DataFrame
        Daily factor history: ``FX:*`` log returns, ``IR:*``/``VOL:*``
        absolute changes.  Must contain every factor in
        ``book.factors(market)``; NaNs raise ``ValueError``.
    alpha : float
        Confidence level in (0, 1).
    horizon_days : float
        VaR horizon; >1 uses sqrt-time scaling of the 1-day figure.
    method : {"plain", "age", "fhs"}
        Plain HS, BRW age-weighted, or filtered HS (EWMA-scaled).
    decay : float
        BRW age-weight decay lambda (only for ``method="age"``).
    ewma_lambda : float
        EWMA decay for FHS devolatilisation (only for ``method="fhs"``).
    min_obs : int
        Minimum history length.
    option_method : str
        Option revaluation passed to ``Book.pnl``.
    warn_pegs : bool
        Emit :class:`PegBlindnessWarning` for near-zero-vol FX factors.

    Returns
    -------
    HistoricalVaRResult
    """
    validate_alpha(alpha)
    validate_horizon(horizon_days)
    factors = book.factors(market)
    if not factors:
        # empty book or base-ccy-cash-only book: zero risk by construction
        return HistoricalVaRResult(0.0, 0.0, alpha, horizon_days, method,
                                   np.zeros(len(returns)), np.full(len(returns), 1.0 / max(len(returns), 1)))
    rets = validate_returns(returns, factors, min_obs=min_obs)
    flagged = tuple(warn_peg_factors(rets, factors)) if warn_pegs else ()

    n = len(rets)
    if method == "plain":
        scen = rets
        weights = np.full(n, 1.0 / n)
    elif method == "age":
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        scen = rets
        ages = np.arange(n - 1, -1, -1)  # last row = age 0
        weights = decay**ages
        weights = weights / weights.sum()
    elif method == "fhs":
        sigma, sigma_next = ewma_volatility(rets, ewma_lambda)
        z = rets.to_numpy() / sigma.to_numpy()
        scen = pd.DataFrame(z * sigma_next.to_numpy()[None, :],
                            index=rets.index, columns=rets.columns)
        weights = np.full(n, 1.0 / n)
    else:
        raise ValueError(f"method must be 'plain', 'age' or 'fhs', got {method!r}")

    pnl = np.asarray(book.pnl(market, scen, option_method=option_method), dtype=float)
    var1, es1 = empirical_var_es(pnl, alpha, weights)
    scale = float(np.sqrt(horizon_days))
    return HistoricalVaRResult(var1 * scale, es1 * scale, alpha, horizon_days,
                               method, pnl, weights, flagged)
