"""Research statistics shared by the alpha, strategy and execution projects.

The alpha, strategy and execution projects import these from here (their old
``research_stats`` modules re-export this one). It holds the tools every
headline number must go through:

* ``sharpe_ratio`` and ``lo_sharpe_se``: the annualised Sharpe ratio and its
  standard error from the Lo (2002) GMM delta method with a Newey-West
  (Bartlett kernel) long-run variance, truncated at a lag window.
* ``deflated_sharpe_ratio``: the probability that the true Sharpe ratio is
  above the expected maximum of ``n_trials`` unskilled trials (Bailey and
  Lopez de Prado, 2014). The trial count comes from a ``TrialLog``.
* ``pbo_cscv``: probability of backtest overfitting by combinatorially
  symmetric cross-validation (Bailey, Borwein, Lopez de Prado, Zhu, 2017).
* ``block_bootstrap_ci``: circular block bootstrap confidence interval for any
  statistic of a return series.
* ``seed_distribution``: mean, standard deviation, standard error and a
  Student-t confidence interval of a statistic computed over many seeds.

Every function validates its input and raises ``ValueError`` on non-finite or
too-short data rather than returning a plausible number.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as _st

FloatArray = NDArray[np.float64]

__all__ = [
    "FloatArray",
    "newey_west_lag",
    "sharpe_ratio",
    "lo_sharpe_se",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "pbo_cscv",
    "block_bootstrap_ci",
    "seed_distribution",
    "SeedSummary",
    "TrialLog",
]

_EULER_GAMMA = 0.5772156649015329


def _as_finite_1d(x: ArrayLike, name: str, min_len: int = 2) -> FloatArray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {a.shape}")
    if a.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} observations, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return a


def _check_periods(periods_per_year: float) -> float:
    p = float(periods_per_year)
    if not math.isfinite(p) or p <= 0:
        raise ValueError(f"periods_per_year must be finite and positive, got {periods_per_year}")
    return p


def newey_west_lag(n: int) -> int:
    """Newey-West (1994) rule-of-thumb truncation lag ``floor(4 (n/100)^(2/9))``."""
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def sharpe_ratio(returns: ArrayLike, periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe ratio ``sqrt(P) * mean / sd`` (sample sd, ddof=1).

    Raises ``ValueError`` when the series has zero variance, because a Sharpe
    ratio is undefined there.
    """
    r = _as_finite_1d(returns, "returns")
    p = _check_periods(periods_per_year)
    sd = r.std(ddof=1)
    if not sd > 0:
        raise ValueError("returns have zero variance; Sharpe ratio is undefined")
    return float(math.sqrt(p) * r.mean() / sd)


def _long_run_variance(psi: FloatArray, lag: int) -> float:
    """Bartlett-kernel (Newey-West) long-run variance of a mean-zero series."""
    n = psi.size
    lrv = float(psi @ psi) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * float(psi[k:] @ psi[:-k]) / n
    return lrv


def lo_sharpe_se(returns: ArrayLike, periods_per_year: float = 252.0, lag: int | None = None) -> float:
    """Standard error of the annualised Sharpe ratio, Lo (2002) with Newey-West.

    The per-period Sharpe ``s = mu / sigma`` has influence function
    ``psi_t = (x_t - mu)/sigma - (s / (2 sigma^2)) ((x_t - mu)^2 - sigma^2)``.
    Its variance is the Bartlett-kernel long-run variance of ``psi`` over ``n``,
    with truncation ``lag`` (default ``newey_west_lag(n)``). With ``lag=0`` and
    Gaussian iid returns this reduces to the textbook ``(1 + s^2/2) / n``.
    The result is scaled by ``sqrt(periods_per_year)``.
    """
    r = _as_finite_1d(returns, "returns", min_len=3)
    p = _check_periods(periods_per_year)
    n = r.size
    if lag is None:
        lag = newey_west_lag(n)
    if lag < 0 or lag >= n:
        raise ValueError(f"lag must be in [0, n-1], got {lag}")
    mu = r.mean()
    var = r.var(ddof=0)
    if not var > 0:
        raise ValueError("returns have zero variance; Sharpe SE is undefined")
    sigma = math.sqrt(var)
    s = mu / sigma
    d = r - mu
    psi = d / sigma - (s / (2.0 * var)) * (d * d - var)
    lrv = max(_long_run_variance(psi, int(lag)), 0.0)
    return float(math.sqrt(p * lrv / n))


def expected_max_sharpe(n_trials: int, trial_sr_variance: float) -> float:
    """Expected maximum of ``n_trials`` per-period Sharpe estimates with true SR 0.

    ``sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))``, with
    ``g`` the Euler-Mascheroni constant (Bailey and Lopez de Prado, 2014).
    ``n_trials = 1`` gives 0.
    """
    if int(n_trials) != n_trials or n_trials < 1:
        raise ValueError(f"n_trials must be a positive integer, got {n_trials}")
    if not (math.isfinite(trial_sr_variance) and trial_sr_variance >= 0):
        raise ValueError(f"trial_sr_variance must be finite and >= 0, got {trial_sr_variance}")
    if n_trials == 1:
        return 0.0
    nrm = _st.norm.ppf
    z = (1.0 - _EULER_GAMMA) * nrm(1.0 - 1.0 / n_trials) + _EULER_GAMMA * nrm(
        1.0 - 1.0 / (n_trials * math.e)
    )
    return float(math.sqrt(trial_sr_variance) * z)


def deflated_sharpe_ratio(
    returns: ArrayLike,
    n_trials: int,
    trial_sharpes: Sequence[float] | None = None,
) -> float:
    """Deflated Sharpe ratio: P(true SR > expected max SR of ``n_trials`` nulls).

    Works on per-period (not annualised) Sharpe ratios. ``trial_sharpes`` are
    the per-period Sharpe ratios of all logged trials; their variance sets the
    spread of the null maximum. Without them the null variance is the sampling
    variance of a zero-SR estimate, ``1/(n-1)``. Skewness and kurtosis of the
    returns enter the denominator as in the probabilistic Sharpe ratio.
    """
    r = _as_finite_1d(returns, "returns", min_len=4)
    n = r.size
    sd = r.std(ddof=1)
    if not sd > 0:
        raise ValueError("returns have zero variance; deflated Sharpe is undefined")
    s = r.mean() / sd
    if trial_sharpes is None:
        v = 1.0 / (n - 1)
    else:
        ts = _as_finite_1d(trial_sharpes, "trial_sharpes", min_len=2)
        v = float(ts.var(ddof=1))
    sr0 = expected_max_sharpe(int(n_trials), v)
    g3 = float(_st.skew(r))
    g4 = float(_st.kurtosis(r, fisher=False))
    denom = 1.0 - g3 * s + (g4 - 1.0) / 4.0 * s * s
    if not denom > 0:
        raise ValueError("non-positive PSR denominator; moments are inconsistent")
    return float(_st.norm.cdf((s - sr0) * math.sqrt(n - 1) / math.sqrt(denom)))


def pbo_cscv(
    performance: ArrayLike,
    n_splits: int = 16,
    metric: Callable[[FloatArray], FloatArray] | None = None,
) -> float:
    """Probability of backtest overfitting by CSCV.

    ``performance`` is a ``T x N`` matrix of per-period returns of ``N``
    configurations. Rows are cut into ``n_splits`` contiguous blocks (even).
    For every way of picking half the blocks as in-sample, the configuration
    with the best in-sample ``metric`` (default mean/sd) is found, its
    out-of-sample rank ``w`` among the N is taken, and the logit
    ``log(w / (1 - w))`` recorded. PBO is the share of logits <= 0, that is the
    share of splits where the in-sample winner ranks at or below the
    out-of-sample median.
    """
    m = np.asarray(performance, dtype=float)
    if m.ndim != 2:
        raise ValueError("performance must be a 2-D array (T x N)")
    t, n_cfg = m.shape
    if n_cfg < 2:
        raise ValueError("need at least two configurations")
    if n_splits < 2 or n_splits % 2:
        raise ValueError(f"n_splits must be an even integer >= 2, got {n_splits}")
    if t < 2 * n_splits:
        raise ValueError(f"need at least {2 * n_splits} rows for {n_splits} splits, got {t}")
    if not np.all(np.isfinite(m)):
        raise ValueError("performance contains NaN or infinite values")

    if metric is None:

        def metric(x: FloatArray) -> FloatArray:
            sd = x.std(axis=0, ddof=1)
            sd = np.where(sd > 0, sd, np.inf)
            return x.mean(axis=0) / sd

    blocks = np.array_split(np.arange(t), n_splits)
    logits: list[float] = []
    for combo in itertools.combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_splits) if i not in combo])
        best = int(np.argmax(metric(m[is_idx])))
        oos_perf = metric(m[oos_idx])
        rank = float(_st.rankdata(oos_perf)[best])  # 1 = worst, N = best
        w = rank / (n_cfg + 1.0)
        logits.append(math.log(w / (1.0 - w)))
    return float(np.mean(np.asarray(logits) <= 0.0))


def block_bootstrap_ci(
    returns: ArrayLike,
    statistic: Callable[[FloatArray], float],
    block_length: int | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Circular block bootstrap percentile CI: ``(point, lower, upper)``.

    Blocks of ``block_length`` consecutive observations (default
    ``ceil(n^(1/3))``) are drawn with wrap-around until ``n`` points are
    filled, which keeps short-range serial dependence inside each block.
    Resamples on which ``statistic`` raises ``ValueError`` (for example a
    zero-variance draw) are dropped; if more than 1% are dropped the call fails.
    """
    r = _as_finite_1d(returns, "returns", min_len=4)
    n = r.size
    if block_length is None:
        block_length = int(math.ceil(n ** (1.0 / 3.0)))
    if not (1 <= block_length <= n):
        raise ValueError(f"block_length must be in [1, n], got {block_length}")
    if n_boot < 100:
        raise ValueError(f"n_boot must be at least 100, got {n_boot}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_length))
    offsets = np.arange(block_length)
    point = float(statistic(r))
    draws: list[float] = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        try:
            draws.append(float(statistic(r[idx])))
        except ValueError:
            continue
    if len(draws) < 0.99 * n_boot:
        raise ValueError("statistic failed on more than 1% of bootstrap resamples")
    lo, hi = np.quantile(np.asarray(draws), [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


@dataclass(frozen=True)
class SeedSummary:
    """Distribution of one statistic across seeds."""

    n: int
    mean: float
    sd: float
    se: float
    ci_low: float
    ci_high: float
    minimum: float
    maximum: float

    def fmt(self, digits: int = 2) -> str:
        f = f"{{:.{digits}f}}"
        return (
            f"{f.format(self.mean)} [{f.format(self.ci_low)}, {f.format(self.ci_high)}]"
            f" (sd {f.format(self.sd)}, n={self.n})"
        )


def seed_distribution(values: ArrayLike, alpha: float = 0.05) -> SeedSummary:
    """Mean, sd, SE and a Student-t CI for the mean of a per-seed statistic."""
    v = _as_finite_1d(values, "values", min_len=2)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = v.size
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    q = float(_st.t.ppf(1.0 - alpha / 2.0, n - 1))
    m = float(v.mean())
    return SeedSummary(n, m, sd, se, m - q * se, m + q * se, float(v.min()), float(v.max()))


@dataclass
class TrialLog:
    """Append-only record of every configuration evaluated on the data.

    The deflated Sharpe ratio is only meaningful with the full trial count,
    including configurations that were looked at and dropped. Log each one.
    """

    trials: list[dict[str, Any]] = field(default_factory=list)

    def log(self, name: str, per_period_sharpe: float | None = None, **params: Any) -> None:
        if per_period_sharpe is not None and not math.isfinite(per_period_sharpe):
            raise ValueError("per_period_sharpe must be finite")
        self.trials.append({"name": name, "sharpe": per_period_sharpe, **params})

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def sharpes(self) -> list[float]:
        return [t["sharpe"] for t in self.trials if t["sharpe"] is not None]
