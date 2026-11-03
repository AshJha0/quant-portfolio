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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as _st

FloatArray = NDArray[np.float64]

__all__ = [
    "Financing",
    "FinancingWarning",
    "FloatArray",
    "SeedSummary",
    "TrialLog",
    "block_bootstrap_ci",
    "deflated_sharpe_detail",
    "deflated_sharpe_from_moments",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "in_market_fraction",
    "lo_sharpe_se",
    "newey_west_lag",
    "pbo_cscv",
    "seed_distribution",
    "sharpe_ratio",
    "warn_no_financing",
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
    return math.floor(float(4.0 * (n / 100.0) ** (2.0 / 9.0)))


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


def in_market_fraction(returns: ArrayLike) -> float:
    """Share of periods with a non-zero return (the strategy held a position)."""
    r = _as_finite_1d(returns, "returns", min_len=1)
    return float(np.count_nonzero(r) / r.size)


def deflated_sharpe_from_moments(sr: float, n_obs: int, skew: float, kurtosis: float,
                                 n_trials: int,
                                 trial_sr_variance: float | None = None) -> dict[str, float]:
    """PSR-form deflated Sharpe from per-period moments (raw kurtosis).

    The same formula as :func:`deflated_sharpe_detail` with ``se_method="psr"``,
    for callers that hold moments rather than the return series.
    """
    if n_obs < 4:
        raise ValueError(f"need at least 4 observations, got {n_obs}")
    v = 1.0 / (n_obs - 1) if trial_sr_variance is None else float(trial_sr_variance)
    sr0 = expected_max_sharpe(int(n_trials), v)
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    if not denom > 0:
        raise ValueError("non-positive PSR denominator; moments are inconsistent")
    z = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return {"sr0_per_period": float(sr0), "dsr": float(_st.norm.cdf(z)), "z": float(z)}


def deflated_sharpe_detail(
    returns: ArrayLike,
    n_trials: int,
    trial_sharpes: Sequence[float] | None = None,
    trial_sr_variance: float | None = None,
    se_method: str = "auto",
) -> dict[str, Any]:
    """Deflated Sharpe ratio with every input it used. This is the one DSR code.

    Plain idea: a best-of-N backtest Sharpe is compared with the Sharpe that
    the best of N skill-less trials would reach by luck. The result is the
    probability that the true Sharpe beats that benchmark.

    Formal statement (Bailey and Lopez de Prado, 2014, eqs. 2-3). Per-period
    Sharpe ``s`` over ``n`` returns, benchmark ``sr0 = expected_max_sharpe(N, V)``:

    * ``se_method="psr"``: ``z = (s - sr0) sqrt(n-1) / sqrt(1 - g3 s + (g4-1)/4 s^2)``;
    * ``se_method="lo"``: ``z = (s - sr0) / se_lo``, with ``se_lo`` the Lo (2002)
      Newey-West standard error of the per-period Sharpe;
    * ``"auto"`` (the default) picks ``"lo"`` when the in-market fraction is
      below 50%. A mostly-zero series has a kurtosis that comes from the zeros,
      and its holding periods are serially dependent (Lo 2002, section 4).

    ``V`` is, in order: ``trial_sr_variance``; the sample variance of the
    per-period ``trial_sharpes``; else ``1/(n-1)``, the sampling variance of a
    zero-Sharpe estimate.

    Worked example: n = 1000, s = 0.05, N = 1, zero skew, kurtosis 3:
    ``sr0 = 0``, ``z = 0.05 sqrt(999) / sqrt(1 + 0.5 * 0.0025) = 1.579``,
    DSR = 0.943.
    """
    if se_method not in ("auto", "psr", "lo"):
        raise ValueError(f"se_method must be 'auto', 'psr' or 'lo', got {se_method!r}")
    r = _as_finite_1d(returns, "returns", min_len=4)
    n = r.size
    sd = r.std(ddof=1)
    if not sd > 0:
        raise ValueError("returns have zero variance; deflated Sharpe is undefined")
    s = float(r.mean() / sd)
    if trial_sr_variance is not None:
        if not (math.isfinite(trial_sr_variance) and trial_sr_variance >= 0):
            raise ValueError("trial_sr_variance must be finite and >= 0")
        v, v_src = float(trial_sr_variance), "given"
    elif trial_sharpes is not None:
        ts = _as_finite_1d(trial_sharpes, "trial_sharpes", min_len=2)
        v, v_src = float(ts.var(ddof=1)), f"variance of {ts.size} trial Sharpes"
    else:
        v, v_src = 1.0 / (n - 1), "1/(n-1), null sampling variance"
    sr0 = expected_max_sharpe(int(n_trials), v)
    g3 = float(_st.skew(r))
    g4 = float(_st.kurtosis(r, fisher=False))
    frac = float(np.count_nonzero(r) / n)
    method = se_method if se_method != "auto" else ("lo" if frac < 0.5 else "psr")
    if method == "psr":
        denom = 1.0 - g3 * s + (g4 - 1.0) / 4.0 * s * s
        if not denom > 0:
            raise ValueError("non-positive PSR denominator; moments are inconsistent")
        se = math.sqrt(denom / (n - 1))
    else:
        se = lo_sharpe_se(r, periods_per_year=1.0)
        if not se > 0:
            raise ValueError("zero Lo standard error; deflated Sharpe is undefined")
    z = (s - sr0) / se
    return {"n_trials": int(n_trials), "trial_sr_variance": v, "trial_sr_variance_source": v_src,
            "sr_per_period": s, "sr0_per_period": float(sr0), "se_per_period": float(se),
            "se_method": method, "in_market_fraction": frac, "skew": g3, "kurtosis": g4,
            "n_obs": int(n), "z": float(z), "dsr": float(_st.norm.cdf(z))}


def deflated_sharpe_ratio(
    returns: ArrayLike,
    n_trials: int,
    trial_sharpes: Sequence[float] | None = None,
    trial_sr_variance: float | None = None,
    se_method: str = "auto",
) -> float:
    """Deflated Sharpe ratio: P(true SR > expected max SR of ``n_trials`` nulls).

    The probability from :func:`deflated_sharpe_detail`; see there for the
    formula, the standard-error choice and where the trial variance comes from.
    """
    return float(deflated_sharpe_detail(returns, n_trials, trial_sharpes, trial_sr_variance,
                                        se_method)["dsr"])


# ---- financing ------------------------------------------------------------

class FinancingWarning(UserWarning):
    """A backtest ran with ``financing=None``: cash earns 0 and borrowing is free."""


@dataclass(frozen=True)
class Financing:
    """Daily financing of a backtest's cash residual.

    Plain idea: a book that holds less than 100% in assets keeps the rest as
    cash, which earns the risk-free rate. A levered book borrows the shortfall
    and pays the rate plus a spread. Leaving this out gives levered books free
    money and understates unlevered ones.

    Formal statement: annualised rate ``r_{t-1}`` (decimal, known at the start
    of period t), accrual ``tau_t`` (calendar days / ``basis`` on a
    DatetimeIndex, else ``1 / periods_per_year``) and cash ``c = 1 - sum(w)``:
    ``fin_t = c (r_{t-1} - lend_spread) tau_t`` if ``c >= 0``, else
    ``c (r_{t-1} + borrow_spread) tau_t``. The risk-free return used for
    excess returns is ``rf_t = r_{t-1} tau_t``.

    Worked example: 150% invested, rate 4%, borrow spread 50 bp, a Friday to
    Monday period on ACT/360: ``fin = -0.5 * 0.045 * 3/360 = -1.875e-4``.
    """

    rate: Any  # float, or a pandas Series of annualised decimal rates by date
    borrow_spread_bp: float = 0.0
    lend_spread_bp: float = 0.0
    basis: float = 360.0
    periods_per_year: float = 252.0

    def __post_init__(self) -> None:
        for name in ("borrow_spread_bp", "lend_spread_bp", "basis", "periods_per_year"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {v}")
        if self.basis <= 0 or self.periods_per_year <= 0:
            raise ValueError("basis and periods_per_year must be positive")

    def accruals(self, index: Any) -> tuple[FloatArray, FloatArray]:
        """(risk-free return, accrual fraction) per period of ``index``; row 0 is 0."""
        import pandas as pd  # local import: the rest of qp_core.stats needs no pandas

        idx = pd.Index(index)
        n = len(idx)
        tau = np.zeros(n)
        if isinstance(idx, pd.DatetimeIndex):
            days = np.diff(idx.to_numpy().astype("datetime64[D]")).astype(float)
            tau[1:] = days / self.basis
        else:
            tau[1:] = 1.0 / self.periods_per_year
        if isinstance(self.rate, pd.Series):
            rate = self.rate.reindex(idx).ffill().to_numpy(dtype=float)
        else:
            rate = np.full(n, float(self.rate))
        if n > 1 and not np.all(np.isfinite(rate[:-1])):
            raise ValueError("financing rate is missing or non-finite on some dates")
        rf = np.zeros(n)
        rf[1:] = rate[:-1] * tau[1:]
        return rf, tau

    def cash_return(self, cash: float, rf_t: float, tau_t: float) -> float:
        """Financing return on cash weight ``cash`` over one period."""
        if cash >= 0:
            return float(cash * (rf_t - self.lend_spread_bp * 1e-4 * tau_t))
        return float(cash * (rf_t + self.borrow_spread_bp * 1e-4 * tau_t))


def warn_no_financing(where: str) -> None:
    """Emit :class:`FinancingWarning` for a backtest run without financing."""
    import warnings

    warnings.warn(f"{where}: financing=None, so cash earns 0 and leverage is free; pass "
                  "qp_core.stats.Financing(rate=...) for excess-return accounting",
                  FinancingWarning, stacklevel=3)


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
        block_length = math.ceil(n ** (1.0 / 3.0))
    if not (1 <= block_length <= n):
        raise ValueError(f"block_length must be in [1, n], got {block_length}")
    if n_boot < 100:
        raise ValueError(f"n_boot must be at least 100, got {n_boot}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_length)
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
