"""VaR / ES backtesting: Kupiec, Christoffersen, Basel traffic light, ES test.

A backtest compares a series of *ex-ante* VaR forecasts against realised
P&L: an **exception** (violation) is a day with ``pnl < -VaR``.

Tests implemented
-----------------
* Kupiec (1995) proportion-of-failures likelihood ratio (unconditional
  coverage), chi-squared with 1 df.
* Christoffersen (1998) independence LR (first-order Markov alternative) and
  the joint conditional-coverage LR (2 df).
* Basel traffic light (spec S4): zone from the cumulative binomial
  probability for any window and alpha. On 250 days at 99 % it is the
  regulatory green (0-4) / yellow (5-9) / red (10+) table.
* ES backtest: Acerbi-Szekely (2014) unconditional Z2 with a simulated
  one-sided p-value (spec S5).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ._validation import finite_array, integer, validate_alpha

__all__ = [
    "exceptions_from_pnl",
    "kupiec_pof",
    "christoffersen_independence",
    "christoffersen_cc",
    "basel_traffic_light",
    "basel_zone_probabilities",
    "acerbi_szekely_z2",
    "exception_cluster_table",
    "rolling_var_backtest",
    "BacktestResult",
]


def exceptions_from_pnl(pnl: np.ndarray, var: np.ndarray | float) -> np.ndarray:
    """Boolean exception indicator: day t is an exception iff ``pnl_t < -VaR_t``.

    ``var`` is the ex-ante VaR (positive for a loss), scalar or per-day array.
    """
    p = finite_array("pnl", pnl, ndim=1, min_size=1)
    v = np.broadcast_to(np.asarray(var, dtype=float), p.shape)
    # NaN fails every comparison, so an unchecked NaN P&L day or NaN VaR would
    # silently record *zero* exceptions and make a broken model pass its
    # Kupiec / Basel backtests green.  Mirrors cpp/equity-var-engine backtest.cpp.
    if not np.isfinite(p).all():
        raise ValueError("pnl contains NaN or infinite values")
    if not np.isfinite(v).all():
        raise ValueError("VaR must be finite")
    if np.any(v < 0):
        raise ValueError("VaR must be reported as a positive loss")
    return p < -v


def _exceptions(exceptions: object) -> np.ndarray:
    """Exception indicators as bool. Refuses NaN/inf and values other than 0/1."""
    arr = np.asarray(exceptions)
    if arr.dtype != bool:
        arr = finite_array("exceptions", arr, ndim=1, min_size=0)
        if not np.isin(arr, (0.0, 1.0)).all():
            raise ValueError("exceptions must be 0/1 or boolean")
    return arr.astype(bool).ravel()


def _lr_pvalue(lr: float, df: int) -> float:
    return float(chi2.sf(lr, df))


def _counts(n_obs: object, n_exceptions: object, alpha: object) -> tuple[int, int, float]:
    n = integer("n_obs", n_obs, min_value=1)
    x = integer("n_exceptions", n_exceptions, min_value=0)
    if x > n:
        raise ValueError(f"n_exceptions must be <= n_obs, got {x} > {n}")
    return n, x, validate_alpha(alpha)


def kupiec_pof(*args: object, n_obs: int | None = None, n_exceptions: int | None = None,
               alpha: float | None = None) -> dict[str, float]:
    """Kupiec proportion-of-failures test (unconditional coverage), spec S3.

    Call as ``kupiec_pof(n_obs=..., n_exceptions=..., alpha=...)``.
    ``LR = -2 ln[(1-a)^(n-x) a^x] + 2 ln[(1-p)^(n-x) p^x]`` with ``p = x/n``
    and ``0 ln 0 = 0``. The p-value is the chi2(1) upper tail. At 250 days
    and 99 % the asymptotic test rejects a correct model about 9.5 % of the
    time at the nominal 5 % level (docs/VALIDATION.md).

    The positional form ``kupiec_pof(n_obs, n_exceptions, alpha=0.01)`` is
    deprecated and still accepted for one release.
    """
    if args:
        if len(args) > 3 or n_obs is not None or n_exceptions is not None:
            raise TypeError("kupiec_pof takes keyword arguments n_obs, n_exceptions, alpha")
        warnings.warn(
            "positional kupiec_pof(n_obs, n_exceptions, alpha) is deprecated; "
            "call kupiec_pof(n_obs=..., n_exceptions=..., alpha=...)",
            DeprecationWarning,
            stacklevel=2,
        )
        n_obs = args[0]
        n_exceptions = args[1] if len(args) > 1 else None
        if len(args) > 2:
            if alpha is not None:
                raise TypeError("alpha given twice")
            alpha = args[2]
        if alpha is None:
            alpha = 0.01
    if n_obs is None or n_exceptions is None or alpha is None:
        raise TypeError("kupiec_pof requires n_obs, n_exceptions and alpha")
    n, x, a = _counts(n_obs, n_exceptions, alpha)
    p = x / n

    def _ll(q: float) -> float:
        out = 0.0
        if n - x > 0 and q < 1.0:
            out += (n - x) * np.log1p(-q)
        if x > 0 and q > 0.0:
            out += x * np.log(q)
        return out

    lr = max(-2.0 * _ll(a) + 2.0 * _ll(p), 0.0)
    return {"lr": float(lr), "pvalue": _lr_pvalue(lr, 1), "expected": a * n, "rate": p}


def christoffersen_independence(exceptions: np.ndarray) -> dict[str, float]:
    """Christoffersen independence LR test against a first-order Markov chain.

    Counts transitions ``n_ij`` (state i yesterday -> j today, 1 = exception)
    and compares the Markov likelihood against the i.i.d. one:
    ``LR_ind ~ chi2(1)`` under independence.  Clustered exceptions (an
    exception today makes one tomorrow more likely) inflate ``n_11`` and
    reject.  Degenerate series (no exceptions, or no state transitions of a
    given kind) use the ``0 ln 0 = 0`` convention.
    """
    ex = _exceptions(exceptions)
    if ex.size < 2:
        raise ValueError(f"need at least 2 observations, got {ex.size}")
    prev, curr = ex[:-1], ex[1:]
    n00 = float(np.sum(~prev & ~curr))
    n01 = float(np.sum(~prev & curr))
    n10 = float(np.sum(prev & ~curr))
    n11 = float(np.sum(prev & curr))

    def _xlogy(a: float, b: float) -> float:
        return a * np.log(b) if a > 0.0 and b > 0.0 else 0.0

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_markov = (
        _xlogy(n00, 1.0 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1.0 - pi11) + _xlogy(n11, pi11)
    )
    ll_iid = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    lr = max(-2.0 * (ll_iid - ll_markov), 0.0)
    return {
        "lr": float(lr),
        "pvalue": _lr_pvalue(lr, 1),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi01": float(pi01),
        "pi11": float(pi11),
    }


def christoffersen_cc(exceptions: np.ndarray, alpha: float = 0.01) -> dict[str, float]:
    """Christoffersen conditional-coverage test: ``LR_cc = LR_uc + LR_ind``.

    Joint test of correct exception rate *and* independence, chi2(2).
    """
    ex = _exceptions(exceptions)
    uc = kupiec_pof(n_obs=ex.size, n_exceptions=int(ex.sum()), alpha=alpha)
    ind = christoffersen_independence(ex)
    lr = uc["lr"] + ind["lr"]
    return {"lr": float(lr), "pvalue": _lr_pvalue(lr, 2), "lr_uc": uc["lr"], "lr_ind": ind["lr"]}


# --------------------------------------------------------------------------- #
# Basel traffic light
# --------------------------------------------------------------------------- #
_YELLOW_ADDON = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def _binom_cdf(x: int, n: int, alpha: float) -> float:
    """``P(X <= x)``, ``X ~ Binomial(n, alpha)``, by pmf recursion in double (S4)."""
    pmf = float(np.exp(n * np.log1p(-alpha)))
    total = pmf
    ratio = alpha / (1.0 - alpha)
    for j in range(x):
        pmf *= (n - j) / (j + 1) * ratio
        total += pmf
    return min(total, 1.0)


def _basel(n: int, x: int, a: float) -> dict[str, object]:
    cum_p = _binom_cdf(x, n, a)
    if cum_p < 0.95:
        zone, plus = "green", 0.0
    elif cum_p < 0.9999:
        zone = "yellow"
        x_eq = next((c for c in range(5, 10) if _binom_cdf(c, 250, 0.01) >= cum_p - 1e-12), 9)
        plus = _YELLOW_ADDON[x_eq]
    else:
        zone, plus = "red", 1.0
    return {"zone": zone, "multiplier": 3.0 + plus, "plus_factor": plus, "cumulative_prob": cum_p}


def basel_traffic_light(*args: object, n_obs: int | None = None, n_exceptions: int | None = None,
                        alpha: float = 0.01) -> dict[str, object]:
    """Basel (1996) traffic light for any window and any alpha (spec S4).

    ``F(x) = P(X <= x)``, ``X ~ Binomial(n_obs, alpha)``. The zone is green
    if ``F < 0.95``, yellow if ``F < 0.9999`` and red otherwise. The yellow
    plus factor is read at the 250-day 99 % count ``x_eq`` in 5..9 with the
    same cumulative probability (smallest ``x'`` with ``F250(x') >= F(x)``,
    else 9). On 250 days at 99 % this is the regulatory table: 0-4 green
    (k = 3), 5-9 yellow (k = 3.40 / 3.50 / 3.65 / 3.75 / 3.85), 10+ red (k = 4).

    Returns ``zone``, ``multiplier`` (k = 3 + plus), ``plus_factor`` and
    ``cumulative_prob``. The positional form ``basel_traffic_light(n_exceptions,
    n_obs=250)`` is deprecated and still accepted for one release.
    """
    if args:
        if len(args) > 2 or n_exceptions is not None:
            raise TypeError("basel_traffic_light takes keyword arguments n_obs, n_exceptions, alpha")
        warnings.warn(
            "positional basel_traffic_light(n_exceptions, n_obs) is deprecated; call "
            "basel_traffic_light(n_obs=..., n_exceptions=..., alpha=...)",
            DeprecationWarning,
            stacklevel=2,
        )
        n_exceptions = args[0]
        if len(args) > 1:
            if n_obs is not None:
                raise TypeError("n_obs given twice")
            n_obs = args[1]
        if n_obs is None:
            n_obs = 250
    if n_obs is None or n_exceptions is None:
        raise TypeError("basel_traffic_light requires n_obs and n_exceptions")
    return _basel(*_counts(n_obs, n_exceptions, alpha))


def basel_zone_probabilities(n_obs: int = 250, alpha: float = 0.01) -> pd.DataFrame:
    """Exact binomial probabilities of each exception count 0..15 and its zone."""
    from scipy.stats import binom

    n_obs = integer("n_obs", n_obs, min_value=1)
    alpha = validate_alpha(alpha)
    rows = []
    for c in range(0, min(15, n_obs) + 1):
        tl = basel_traffic_light(n_obs=n_obs, n_exceptions=c, alpha=alpha)
        rows.append(
            {
                "exceptions": c,
                "zone": tl["zone"],
                "multiplier": tl["multiplier"],
                "prob_exact": float(binom.pmf(c, n_obs, alpha)),
                "prob_cumulative": tl["cumulative_prob"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ES backtest (Acerbi-Szekely Z2, unconditional)
# --------------------------------------------------------------------------- #
def _z2(p: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float) -> np.ndarray:
    """Z2 along the last axis: ``sum_t pnl_t I_t / (T alpha ES_t) + 1``."""
    hits = p < -v
    return np.sum(np.where(hits, p / e, 0.0), axis=-1) / (alpha * p.shape[-1]) + 1.0


def acerbi_szekely_z2(
    pnl: np.ndarray,
    var: np.ndarray | float,
    es: np.ndarray | float,
    *args: object,
    alpha: float | None = None,
    n_sims: int = 10_000,
    seed: int | np.random.Generator | None = 0,
    null_sampler=None,
) -> dict[str, float]:
    """Acerbi-Szekely (2014) unconditional ES backtest Z2 with a simulated p-value (spec S5).

    ``Z2 = sum_t pnl_t I_t / (T alpha ES_t) + 1``, ``I_t = 1{pnl_t < -VaR_t}``.
    ``E[Z2] = 0`` under the null. A NEGATIVE value means realised tail losses
    exceed the forecast ES (ES understated).

    The p-value is one-sided, ``P(Z2_sim <= Z2_obs)``, from ``n_sims`` paths
    simulated under the null with the same ``VaR_t`` and ``ES_t``. The default
    null is ``pnl_t ~ Normal(0, sigma_t)`` with ``sigma_t = ES_t * alpha /
    phi(z)``, ``z = Phi^{-1}(1 - alpha)``. Pass ``null_sampler(rng, n_sims, T)``
    returning an ``(n_sims, T)`` array to use the model's own distribution.

    Returns ``z2``, ``p_value`` and ``n_exceptions``. Passing ``alpha``
    positionally is deprecated. ``alpha`` defaults to 0.025.
    """
    from scipy.stats import norm

    if args:
        if len(args) > 1 or alpha is not None:
            raise TypeError("acerbi_szekely_z2 takes alpha as a keyword argument")
        warnings.warn(
            "positional alpha in acerbi_szekely_z2 is deprecated; pass alpha=...",
            DeprecationWarning,
            stacklevel=2,
        )
        alpha = args[0]
    if alpha is None:
        alpha = 0.025
    alpha = validate_alpha(alpha)
    n_sims = integer("n_sims", n_sims, min_value=1)
    p = finite_array("pnl", pnl, ndim=1, min_size=1)
    v = np.broadcast_to(finite_array("VaR", var, min_size=1), p.shape).astype(float)
    e = np.broadcast_to(finite_array("ES", es, min_size=1), p.shape).astype(float)
    if np.any(v < 0):
        raise ValueError("VaR must be reported as a positive loss")
    if np.any(e <= 0):
        raise ValueError("ES must be reported as a positive loss")
    if np.any(e + 1e-12 * np.maximum(1.0, np.abs(e)) < v):
        raise ValueError("ES must be >= VaR on every day")
    z2 = float(_z2(p, v, e, alpha))
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    t = p.size
    sigma = e * alpha / norm.pdf(norm.ppf(1.0 - alpha))
    chunk = max(1, min(n_sims, 2_000_000 // t))
    count, done = 0, 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        if null_sampler is None:
            sims = rng.standard_normal((m, t)) * sigma
        else:
            sims = np.asarray(null_sampler(rng, m, t), dtype=float)
            if sims.shape != (m, t) or not np.isfinite(sims).all():
                raise ValueError(f"null_sampler must return a finite ({m}, {t}) array")
        count += int(np.sum(_z2(sims, v, e, alpha) <= z2))
        done += m
    return {"z2": z2, "p_value": count / n_sims, "n_exceptions": int(np.sum(p < -v))}


# --------------------------------------------------------------------------- #
# Rolling backtest driver + reporting
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    """Container for one method's rolling backtest over a P&L history."""

    name: str
    exceptions: np.ndarray
    var_series: np.ndarray
    pnl_series: np.ndarray
    alpha: float

    @property
    def n_obs(self) -> int:
        return int(self.exceptions.size)

    @property
    def n_exceptions(self) -> int:
        return int(self.exceptions.sum())

    def summary(self) -> dict[str, object]:
        uc = kupiec_pof(n_obs=self.n_obs, n_exceptions=self.n_exceptions, alpha=self.alpha)
        ind = christoffersen_independence(self.exceptions)
        cc = christoffersen_cc(self.exceptions, self.alpha)
        out: dict[str, object] = {
            "method": self.name,
            "n_obs": self.n_obs,
            "exceptions": self.n_exceptions,
            "expected": uc["expected"],
            "kupiec_lr": uc["lr"],
            "kupiec_p": uc["pvalue"],
            "indep_p": ind["pvalue"],
            "cc_p": cc["pvalue"],
        }
        out["basel_zone"] = basel_traffic_light(
            n_obs=self.n_obs, n_exceptions=self.n_exceptions, alpha=self.alpha
        )["zone"]
        return out


def rolling_var_backtest(
    pnl: np.ndarray,
    var_fn,
    window: int = 250,
    alpha: float = 0.01,
    name: str = "model",
) -> BacktestResult:
    """Walk-forward backtest: forecast VaR for day t from ``pnl[t-window:t]``.

    Parameters
    ----------
    pnl : array
        Daily P&L history (currency units).
    var_fn : callable ``(history, alpha) -> float``
        VaR estimator applied to each trailing window (positive-loss output).
    window : int
        Estimation window length in days.
    """
    p = finite_array("pnl", pnl, ndim=1, min_size=1)
    window = integer("window", window, min_value=1)
    alpha = validate_alpha(alpha)
    if p.size <= window:
        raise ValueError(
            f"need more than window={window} observations to backtest, got {p.size}"
        )
    # Refuse a poisoned feed *before* any VaR is computed: a single NaN would
    # otherwise propagate into every trailing-window forecast.
    if not np.isfinite(p).all():
        raise ValueError("pnl contains NaN or infinite values")
    n_fore = p.size - window
    var_series = np.empty(n_fore)
    for i in range(n_fore):
        var_series[i] = var_fn(p[i : i + window], alpha)
    realised = p[window:]
    ex = exceptions_from_pnl(realised, var_series)
    return BacktestResult(name, ex, var_series, realised, alpha)


def exception_cluster_table(exceptions: np.ndarray) -> pd.DataFrame:
    """Table of exception days with gaps — a visual clustering diagnostic.

    Columns: exception day index, gap since the previous exception, and a
    ``clustered`` flag (gap <= 5 business days).  Under a correct i.i.d.
    model at 99 %, gaps are geometric with mean ~100 days; a run of small
    gaps is the visual signature Christoffersen's LR formalises.
    """
    ex = _exceptions(exceptions)
    days = np.flatnonzero(ex)
    gaps = np.diff(days, prepend=-1)  # first gap: distance from the start of the window
    clustered = np.concatenate([[False], np.diff(days) <= 5]) if days.size else np.zeros(0, bool)
    return pd.DataFrame({"day": days, "gap_days": gaps, "clustered": clustered})
