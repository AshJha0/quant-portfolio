"""VaR / ES backtesting: Kupiec, Christoffersen, Basel traffic light, ES test.

A backtest compares a series of *ex-ante* VaR forecasts against realised
P&L: an **exception** (violation) is a day with ``pnl < -VaR``.

Tests implemented
-----------------
* Kupiec (1995) proportion-of-failures likelihood ratio (unconditional
  coverage), chi-squared with 1 df.
* Christoffersen (1998) independence LR (first-order Markov alternative) and
  the joint conditional-coverage LR (2 df).
* Basel traffic light: 250-day, 99 % VaR exception count mapped to
  green (0-4) / yellow (5-9) / red (10+) with the regulatory multiplier
  add-on schedule; the binomial exceedance probabilities behind the zone
  boundaries are exposed for documentation.
* ES backtest: Acerbi-Szekely (2014) unconditional Z2 statistic (chosen over
  the severity z-test because it uses every exception's magnitude relative
  to the ex-ante ES and needs no normality assumption; see METHODOLOGY.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2

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
    p = np.asarray(pnl, dtype=float).ravel()
    v = np.broadcast_to(np.asarray(var, dtype=float), p.shape)
    if np.any(v < 0):
        raise ValueError("VaR must be reported as a positive loss")
    return p < -v


def _lr_pvalue(lr: float, df: int) -> float:
    return float(chi2.sf(lr, df))


def kupiec_pof(n_obs: int, n_exceptions: int, alpha: float = 0.01) -> dict[str, float]:
    """Kupiec proportion-of-failures test (unconditional coverage).

    ``LR_uc = -2 ln[ (1-p)^{T-x} p^x / ((1-x/T)^{T-x} (x/T)^x) ]`` with
    ``p = alpha``, ``T = n_obs``, ``x = n_exceptions``; asymptotically
    chi2(1) under H0 (exception probability = alpha).  ``x = 0`` and
    ``x = T`` are handled by the ``0 * ln 0 = 0`` convention.

    Returns
    -------
    dict with ``lr`` (statistic), ``pvalue``, ``expected`` exceptions and the
    observed exception ``rate``.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    if not 0 <= n_exceptions <= n_obs:
        raise ValueError(f"n_exceptions must be in [0, {n_obs}], got {n_exceptions}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    t, x = float(n_obs), float(n_exceptions)
    pihat = x / t

    def _ll(p: float) -> float:
        out = 0.0
        if t - x > 0:
            out += (t - x) * np.log(1.0 - p)
        if x > 0:
            out += x * np.log(p)
        return out

    lr = -2.0 * (_ll(alpha) - _ll(pihat)) if pihat not in (0.0, 1.0) else -2.0 * (
        _ll(alpha) - 0.0
    )
    lr = max(float(lr), 0.0)
    return {
        "lr": lr,
        "pvalue": _lr_pvalue(lr, 1),
        "expected": alpha * t,
        "rate": pihat,
    }


def christoffersen_independence(exceptions: np.ndarray) -> dict[str, float]:
    """Christoffersen independence LR test against a first-order Markov chain.

    Counts transitions ``n_ij`` (state i yesterday -> j today, 1 = exception)
    and compares the Markov likelihood against the i.i.d. one:
    ``LR_ind ~ chi2(1)`` under independence.  Clustered exceptions (an
    exception today makes one tomorrow more likely) inflate ``n_11`` and
    reject.  Degenerate series (no exceptions, or no state transitions of a
    given kind) use the ``0 ln 0 = 0`` convention.
    """
    ex = np.asarray(exceptions).astype(bool).ravel()
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
    ex = np.asarray(exceptions).astype(bool).ravel()
    uc = kupiec_pof(ex.size, int(ex.sum()), alpha)
    ind = christoffersen_independence(ex)
    lr = uc["lr"] + ind["lr"]
    return {"lr": float(lr), "pvalue": _lr_pvalue(lr, 2), "lr_uc": uc["lr"], "lr_ind": ind["lr"]}


# --------------------------------------------------------------------------- #
# Basel traffic light
# --------------------------------------------------------------------------- #
_YELLOW_ADDON = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def basel_traffic_light(n_exceptions: int, n_obs: int = 250) -> dict[str, object]:
    """Basel (1996 supervisory framework) traffic-light zone for 99 % VaR.

    On the standard 250-day window: 0-4 exceptions = green (multiplier 3.0),
    5-9 = yellow (add-ons 0.40/0.50/0.65/0.75/0.85), 10+ = red (add-on 1.0,
    multiplier 4.0, presumption of a flawed model).  The zone boundaries come
    from the exact Binomial(250, 0.01) distribution: green covers ~89 % of
    outcomes for a correct model, red has ~0.03 % probability
    (see ``basel_zone_probabilities``).

    Returns dict with ``zone``, ``multiplier`` (k = 3 + add-on) and the
    cumulative binomial probability of seeing <= n_exceptions under a
    correct model.
    """
    if n_exceptions < 0:
        raise ValueError(f"n_exceptions must be >= 0, got {n_exceptions}")
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}")
    if n_exceptions <= 4:
        zone, addon = "green", 0.0
    elif n_exceptions <= 9:
        zone, addon = "yellow", _YELLOW_ADDON[n_exceptions]
    else:
        zone, addon = "red", 1.0
    from scipy.stats import binom

    cum_p = float(binom.cdf(min(n_exceptions, n_obs), n_obs, 0.01))
    return {"zone": zone, "multiplier": 3.0 + addon, "cumulative_prob": cum_p}


def basel_zone_probabilities(n_obs: int = 250, alpha: float = 0.01) -> pd.DataFrame:
    """Exact binomial probabilities of each exception count / zone boundary."""
    from scipy.stats import binom

    counts = np.arange(0, 16)
    rows = []
    for c in counts:
        tl = basel_traffic_light(int(c), n_obs)
        rows.append(
            {
                "exceptions": int(c),
                "zone": tl["zone"],
                "multiplier": tl["multiplier"],
                "prob_exact": float(binom.pmf(c, n_obs, alpha)),
                "prob_cumulative": float(binom.cdf(c, n_obs, alpha)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ES backtest (Acerbi-Szekely Z2, unconditional)
# --------------------------------------------------------------------------- #
def acerbi_szekely_z2(
    pnl: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    alpha: float = 0.025,
) -> dict[str, float]:
    """Acerbi-Szekely (2014) unconditional ES backtest statistic Z2.

    ``Z2 = (1/T) * sum_t [ pnl_t * I(pnl_t < -VaR_t) / (alpha * ES_t) ] + 1``

    Under H0 (P&L drawn from the model's distribution) ``E[Z2] = 0``; a
    materially **negative** Z2 means realised tail losses exceed the model
    ES — the model understates tail risk.  Acerbi-Szekely report that the
    5 % critical value is ~ -0.70 across realistic P&L distributions, so we
    flag ``reject = Z2 < -0.70`` (documented approximation; an exact p-value
    requires simulating under the model, docs/METHODOLOGY.md).

    Chosen over the exception-severity z-test because it uses full exception
    magnitudes without assuming normal tails.
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
    p = np.asarray(pnl, dtype=float).ravel()
    v = np.broadcast_to(np.asarray(var, dtype=float), p.shape).astype(float)
    e = np.broadcast_to(np.asarray(es, dtype=float), p.shape).astype(float)
    if np.any(e <= 0):
        raise ValueError("ES must be reported as a positive loss")
    if np.any(e + 1e-12 < v):
        raise ValueError("ES must be >= VaR on every day")
    hits = p < -v
    z2 = float(np.sum(p[hits] / e[hits]) / (alpha * p.size) + 1.0) if hits.any() else 1.0
    return {"z2": z2, "n_exceptions": int(hits.sum()), "reject": bool(z2 < -0.70)}


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
        uc = kupiec_pof(self.n_obs, self.n_exceptions, self.alpha)
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
        if self.alpha == 0.01:
            scaled = int(round(self.n_exceptions * 250 / max(self.n_obs, 1)))
            out["basel_zone_250d"] = basel_traffic_light(scaled)["zone"]
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
    p = np.asarray(pnl, dtype=float).ravel()
    if p.size <= window:
        raise ValueError(
            f"need more than window={window} observations to backtest, got {p.size}"
        )
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
    ex = np.asarray(exceptions).astype(bool).ravel()
    days = np.flatnonzero(ex)
    gaps = np.diff(days, prepend=days[0] if days.size else 0)
    if days.size:
        gaps[0] = days[0] + 1  # distance from start of window
    return pd.DataFrame(
        {
            "day": days,
            "gap_days": gaps,
            "clustered": np.concatenate([[False], np.diff(days) <= 5]) if days.size else [],
        }
    )
