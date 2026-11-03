"""VaR/ES backtesting: Kupiec, Christoffersen, Basel traffic light, ES test.

Exception convention: day t is an exception when the realised **loss**
exceeds the VaR forecast made for day t: ``-pnl_t > var_t``.

Tests
-----
All functions take ``alpha`` as the tail probability (0.01 = 99 % VaR).

* Kupiec POF (unconditional coverage): LR test that the exception
  frequency equals ``alpha``; chi2(1).
* Christoffersen independence: LR test against first-order Markov
  clustering of exceptions; chi2(1).  FX desks care because volatility
  clustering makes an unconditional method (plain HS, sample-cov
  parametric) fail *this* test long before it fails Kupiec.
* Conditional coverage: LR_cc = LR_uc + LR_ind; chi2(2).
* Basel traffic light: zones from ``F(x) = P(X <= x)``,
  ``X ~ Binomial(n_obs, alpha)``: green if ``F < 0.95``, yellow if
  ``F < 0.9999``, red otherwise.  For 250 days at 1 % this gives green
  0-4, yellow 5-9, red 10+.  The yellow plus factor for any window is the
  1996 table entry of the 250-day count with the same cumulative
  probability (see :func:`basel_traffic_light`).
* ES backtest: Acerbi-Szekely (2014) test 2 in the paper's sign,
  ``Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1``, ``I_t = 1{pnl_t < -VaR_t}``.
  Negative Z2 means ES is understated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .book import Book, Market
from .common import resolve_alpha, validate_alpha, validate_finite

__all__ = [
    "BacktestResult",
    "TrafficLight",
    "acerbi_szekely_z2",
    "basel_traffic_light",
    "christoffersen_independence",
    "conditional_coverage",
    "evaluate_var_backtest",
    "kupiec_pof",
    "rolling_backtest",
]

#: 1996 Basel yellow-zone capital multiplier add-ons by exception count
#: (250-day, 99% window); multiplier = 3.0 + add-on, capped at 4.0 in red.
BASEL_ADDONS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def _xlogy(x: float, y: float) -> float:
    """x*log(y) with the 0*log(0) = 0 convention."""
    if x == 0:
        return 0.0
    if y <= 0:
        return -np.inf
    return x * np.log(y)


def _counts(n_obs, n_exceptions) -> tuple[int, int]:
    validate_finite(n_obs=n_obs, n_exceptions=n_exceptions)
    if int(n_obs) != n_obs or int(n_exceptions) != n_exceptions:
        raise ValueError("n_obs and n_exceptions must be integers")
    n, x = int(n_obs), int(n_exceptions)
    if n < 1:
        raise ValueError(f"n_obs must be >= 1, got {n}")
    if x < 0:
        raise ValueError(f"n_exceptions must be >= 0, got {x}")
    if x > n:
        raise ValueError(f"n_exceptions ({x}) must not exceed n_obs ({n})")
    return n, x


def _kupiec(n: int, x: int, alpha: float) -> tuple[float, float]:
    pi_hat = x / n
    ll0 = _xlogy(n - x, 1.0 - alpha) + _xlogy(x, alpha)
    ll1 = _xlogy(n - x, 1.0 - pi_hat) + _xlogy(x, pi_hat)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return float(lr), float(chi2.sf(lr, df=1))


def kupiec_pof(*, n_obs: int, n_exceptions: int, alpha: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures LR test.

    Call as ``kupiec_pof(n_obs=..., n_exceptions=..., alpha=...)`` with
    ``alpha`` the tail probability.  Returns ``(LR_uc, p_value)``, LR_uc
    ~ chi2(1) under H0 that the exception probability is ``alpha``.

    The old positional form ``kupiec_pof(n_exceptions, n_obs, confidence)``
    was removed in v3.0.0.

    Raises
    ------
    ValueError
        ``alpha`` outside (0, 0.5), ``n_obs < 1``, ``n_exceptions < 0`` or
        ``n_exceptions > n_obs``.
    """
    alpha = validate_alpha(alpha)
    n, x = _counts(n_obs, n_exceptions)
    return _kupiec(n, x, alpha)


def _exceedance_array(exceedances) -> np.ndarray:
    raw = np.asarray(exceedances, dtype=float).ravel()
    if raw.size < 2:
        raise ValueError("need at least 2 observations for the independence test")
    if not np.isfinite(raw).all() or not np.isin(raw, (0.0, 1.0)).all():
        raise ValueError("exceedances must be a 0/1 (or boolean) series")
    return raw.astype(int)


def christoffersen_independence(exceedances) -> tuple[float, float]:
    """Christoffersen (1998) independence LR test on a 0/1 exception series.

    Tests first-order Markov dependence: H0 says P(exception | exception
    yesterday) = P(exception | none yesterday).  Returns ``(LR_ind, p)``,
    chi2(1).  Degenerate series (no transitions of one kind) return LR=0.
    """
    e = _exceedance_array(exceedances)
    prev, curr = e[:-1], e[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    ll1 = (_xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01)
           + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11))
    lr = -2.0 * (ll0 - ll1)
    lr = max(lr, 0.0) if np.isfinite(lr) else 0.0
    return float(lr), float(chi2.sf(lr, df=1))


def conditional_coverage(exceedances, alpha: float | None = None) -> tuple[float, float]:
    """Christoffersen conditional coverage: LR_cc = LR_uc + LR_ind, chi2(2).

    ``alpha`` is the tail probability (default 0.01).
    """
    alpha = resolve_alpha(alpha)
    e = _exceedance_array(exceedances)
    lr_uc, _ = kupiec_pof(n_obs=e.size, n_exceptions=int(e.sum()), alpha=alpha)
    lr_ind, _ = christoffersen_independence(e)
    lr_cc = lr_uc + lr_ind
    return float(lr_cc), float(chi2.sf(lr_cc, df=2))


@dataclass(frozen=True)
class TrafficLight:
    """Basel traffic-light outcome for a backtest window."""

    zone: str  # "green" | "yellow" | "red"
    n_exceptions: int
    n_obs: int
    cumulative_prob: float  # P(X <= n_exceptions), X ~ Binomial(n_obs, alpha)
    multiplier: float  # capital multiplier k = 3 + plus_factor
    plus_factor: float = 0.0


def _binom_cdf(x: int, n: int, p: float) -> float:
    """``P(X <= x)`` for Binomial(n, p) by pmf recursion in double."""
    pmf = float(np.exp(n * np.log1p(-p)))
    cdf = pmf
    ratio = p / (1.0 - p)
    for j in range(x):
        pmf *= (n - j) / (j + 1) * ratio
        cdf += pmf
    return min(cdf, 1.0)


_F250 = {x: _binom_cdf(x, 250, 0.01) for x in range(5, 10)}


def _traffic_light(n: int, x: int, alpha: float) -> TrafficLight:
    cum = _binom_cdf(x, n, alpha)
    if cum < 0.95:
        return TrafficLight("green", x, n, cum, 3.0, 0.0)
    if cum < 0.9999:
        x_eq = next((k for k in range(5, 10) if _F250[k] >= cum - 1e-12), 9)
        plus = BASEL_ADDONS[x_eq]
        return TrafficLight("yellow", x, n, cum, 3.0 + plus, plus)
    return TrafficLight("red", x, n, cum, 4.0, 1.0)


def basel_traffic_light(*, n_obs: int, n_exceptions: int, alpha: float | None = None) -> TrafficLight:
    """Basel traffic-light zone and capital multiplier for any window.

    Call as ``basel_traffic_light(n_obs=..., n_exceptions=..., alpha=0.01)``.
    ``F(x) = P(X <= x)`` with ``X ~ Binomial(n_obs, alpha)``: green if
    ``F < 0.95``, yellow if ``F < 0.9999``, red otherwise.  Plus factor:
    green 0, red 1.0.  Yellow uses the smallest 250-day count ``x'`` in
    5..9 whose ``Binomial(250, 0.01)`` CDF reaches ``F(x)`` (9 if none) and
    takes the 1996 add-on for ``x'`` (0.40, 0.50, 0.65, 0.75, 0.85).
    Multiplier = 3 + plus factor.  For 250 days at 1 % this is the
    regulatory table.

    The old positional form ``basel_traffic_light(n_exceptions, n_obs=250,
    confidence=0.99)`` was removed in v3.0.0.
    """
    alpha = validate_alpha(0.01 if alpha is None else alpha)
    n, x = _counts(n_obs, n_exceptions)
    return _traffic_light(n, x, alpha)


@dataclass(frozen=True)
class BacktestResult:
    """Full VaR backtest summary over a forecast/realisation window."""

    n_obs: int
    n_exceptions: int
    exception_rate: float
    expected_rate: float
    kupiec_lr: float
    kupiec_p: float
    independence_lr: float
    independence_p: float
    cc_lr: float
    cc_p: float
    traffic_light: TrafficLight
    exceedances: np.ndarray


def evaluate_var_backtest(pnl, var_forecasts, alpha: float | None = None) -> BacktestResult:
    """Score a realised P&L series against out-of-sample VaR forecasts.

    Parameters
    ----------
    pnl : array_like
        Realised P&L (profit +) per day.
    var_forecasts : array_like
        Positive VaR forecast for the same day (made ex ante).
    alpha : float
        Tail probability of the VaR forecasts (default 0.01).
    """
    alpha = resolve_alpha(alpha)
    pnl = np.asarray(pnl, dtype=float).ravel()
    var = np.asarray(var_forecasts, dtype=float).ravel()
    if pnl.size != var.size:
        raise ValueError("pnl and var_forecasts must have equal length")
    if pnl.size < 2:
        raise ValueError("need at least 2 observations to backtest")
    if not (np.isfinite(pnl).all() and np.isfinite(var).all()):
        raise ValueError(
            "backtest inputs contain NaN or infinite values (NaN policy: refuse)"
        )
    exc = (-pnl > var).astype(int)
    x = int(exc.sum())
    lr_uc, p_uc = kupiec_pof(n_obs=exc.size, n_exceptions=x, alpha=alpha)
    lr_ind, p_ind = christoffersen_independence(exc)
    lr_cc = lr_uc + lr_ind
    p_cc = float(chi2.sf(lr_cc, df=2))
    tl = basel_traffic_light(n_obs=exc.size, n_exceptions=x, alpha=alpha)
    return BacktestResult(exc.size, x, x / exc.size, alpha, lr_uc, p_uc,
                          lr_ind, p_ind, lr_cc, p_cc, tl, exc)


def rolling_backtest(
    book: Book,
    market: Market,
    returns: pd.DataFrame,
    var_fn: Callable[[Book, Market, pd.DataFrame], float],
    window: int = 250,
    alpha: float | None = None,
    option_method: str = "full",
) -> pd.DataFrame:
    """Rolling out-of-sample VaR backtest on a factor-return history.

    For each day t >= window, ``var_fn(book, market, returns[t-window:t])``
    produces the ex-ante VaR, and the realised (hypothetical) P&L is the
    full revaluation of today's book under day t's factor returns - the
    standard static-book backtest P&L (see docs/DESK_GUIDE.md).

    Returns
    -------
    pandas.DataFrame
        Columns ``pnl``, ``var``, ``exceed`` indexed like ``returns``
        (first ``window`` days dropped).
    """
    resolve_alpha(alpha)  # validated for API symmetry; var_fn owns alpha
    if window < 30:
        raise ValueError("window must be >= 30")
    if len(returns) <= window:
        raise ValueError(
            f"need more than window={window} rows of history, got {len(returns)}"
        )
    factors = book.factors(market)
    rows = []
    idx = []
    pnl_all = np.asarray(
        book.pnl(market, returns[factors], option_method=option_method), dtype=float
    )
    for t in range(window, len(returns)):
        win = returns.iloc[t - window : t]
        v = float(var_fn(book, market, win))
        realised = float(pnl_all[t])
        rows.append((realised, v, float(-realised > v)))
        idx.append(returns.index[t])
    out = pd.DataFrame(rows, columns=["pnl", "var", "exceed"], index=idx)
    return out


def _as_z2_inputs(pnl, var, es):
    pnl = np.asarray(pnl, dtype=float).ravel()
    var = np.asarray(var, dtype=float).ravel()
    es = np.asarray(es, dtype=float).ravel()
    if pnl.size == 0:
        raise ValueError("pnl is empty")
    if not (pnl.size == var.size == es.size):
        raise ValueError("pnl, var and es must have equal length")
    if not (np.isfinite(pnl).all() and np.isfinite(var).all()
            and np.isfinite(es).all()):
        raise ValueError(
            "pnl/var/es contain NaN or infinite values (NaN policy: refuse)"
        )
    if np.any(es <= 0):
        raise ValueError("es forecasts must be positive losses")
    return pnl, var, es


def _z2(pnl: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    """Z2 along the last axis; ``pnl`` may be (T,) or (n_sims, T)."""
    t = pnl.shape[-1]
    hit = pnl < -var
    return np.sum(pnl * hit / es, axis=-1) / (t * alpha) + 1.0


def acerbi_szekely_z2(pnl, var, es, *, alpha: float, n_sims: int = 10_000,
                      seed: int | np.random.Generator = 0,
                      null_sampler: Callable | None = None) -> dict:
    """Acerbi-Szekely (2014) ES test 2, in the paper's sign.

    ``Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1``, ``I_t = 1{pnl_t < -VaR_t}``.
    ``E[Z2] = 0`` under a correct model.  Negative Z2 means ES is
    understated.

    The one-sided p-value ``P(Z2_sim <= Z2_obs)`` is simulated under the
    null.  By default ``pnl_t ~ Normal(0, sigma_t)`` with
    ``sigma_t = ES_t / (phi(z)/alpha)``, ``z = Phi^{-1}(1 - alpha)``.  The
    simulated statistic uses the same ``var`` and ``es`` forecasts as the
    observed one.  Pass ``null_sampler(rng, n) -> array (n, T)`` to draw
    P&L paths from the model's own distribution instead.

    Returns
    -------
    dict
        ``{"z2", "p_value", "n_exceptions"}``.
    """
    alpha = validate_alpha(alpha)
    pnl, var, es = _as_z2_inputs(pnl, var, es)
    if int(n_sims) != n_sims or n_sims < 1:
        raise ValueError(f"n_sims must be a positive integer, got {n_sims}")
    n_sims = int(n_sims)
    z_obs = float(_z2(pnl, var, es, alpha))
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    t = pnl.size
    z = norm.isf(alpha)
    sig = es / (norm.pdf(z) / alpha)
    chunk = max(1, (1 << 20) // t)
    below = 0
    done = 0
    while done < n_sims:
        r = min(chunk, n_sims - done)
        if null_sampler is None:
            sims = rng.standard_normal((r, t)) * sig[None, :]
        else:
            sims = np.asarray(null_sampler(rng, r), dtype=float)
            if sims.shape != (r, t) or not np.isfinite(sims).all():
                raise ValueError(f"null_sampler must return a finite array of shape ({r}, {t})")
        below += int(np.sum(_z2(sims, var[None, :], es[None, :], alpha) <= z_obs))
        done += r
    p = (below + 1) / (n_sims + 1)
    return {"z2": z_obs, "p_value": float(p), "n_exceptions": int(np.sum(pnl < -var))}
