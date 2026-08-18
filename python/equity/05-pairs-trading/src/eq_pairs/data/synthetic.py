"""Seeded synthetic price generators for pairs-trading research and tests.

All generators are deterministic given a seed (or an explicit
``numpy.random.Generator``) and produce **daily close prices** on a business-day
index. Conventions:

* Prices are in dollars, strictly positive (generators work in arithmetic
  price space with small increments around a base of ~100, and floor prices
  well above zero).
* Time is measured in trading days; OU parameters (``kappa``) are per day, so
  half-life = ln(2)/kappa is in days.
* Every generator returns the *true* data-generating parameters alongside the
  prices so tests can assert parameter recovery.

Three canonical regimes are provided:

(a) :func:`cointegrated_pair` — true cointegration with known hedge ratio and
    a known Ornstein-Uhlenbeck spread (kappa, sigma known).
(b) :func:`correlated_random_walks` — the classic trap: two *independent*
    random walks whose daily increments are highly correlated. Returns are
    correlated, price levels drift apart, and there is **no** cointegration.
(c) :func:`regime_break_pair` — cointegrated in the first part of the sample,
    after which the spread stops mean-reverting (cointegration dies).
(d) :func:`mixed_panel` — a sector-tagged panel mixing all three plus
    idiosyncratic names, used by the selection-funnel example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "PairTruth",
    "PanelTruth",
    "make_rng",
    "business_index",
    "simulate_ou",
    "cointegrated_pair",
    "correlated_random_walks",
    "regime_break_pair",
    "mixed_panel",
]

SeedLike = Union[int, np.random.Generator, None]

_PRICE_FLOOR = 1.0  # dollars; generators floor prices here to stay positive


def make_rng(seed: SeedLike = None) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from a seed or pass one through."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def business_index(n: int, start: str = "2015-01-01") -> pd.DatetimeIndex:
    """Business-day DatetimeIndex of length ``n`` starting at ``start``."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return pd.bdate_range(start=start, periods=n)


@dataclass(frozen=True)
class PairTruth:
    """True data-generating parameters of a synthetic pair.

    Attributes
    ----------
    kind : str
        One of ``{"cointegrated", "correlated_rw", "regime_break"}``.
    beta : float
        True hedge ratio (units of leg-x per unit of leg-y); NaN when the
        pair is not cointegrated.
    alpha : float
        True intercept of the cointegrating relation (dollars).
    kappa : float
        OU mean-reversion speed per day (NaN if no OU spread).
    sigma : float
        OU diffusion volatility in dollars per sqrt(day) (NaN if none).
    mu : float
        OU long-run spread mean in dollars.
    half_life : float
        ln(2)/kappa in trading days (inf/NaN when not mean-reverting).
    break_index : int
        Sample index at which cointegration dies (-1 if never).
    """

    kind: str
    beta: float = np.nan
    alpha: float = 0.0
    kappa: float = np.nan
    sigma: float = np.nan
    mu: float = 0.0
    half_life: float = np.nan
    break_index: int = -1


@dataclass
class PanelTruth:
    """Ground truth for :func:`mixed_panel`."""

    sectors: dict[str, str] = field(default_factory=dict)
    pairs: dict[tuple[str, str], PairTruth] = field(default_factory=dict)

    def cointegrated_pairs(self) -> list[tuple[str, str]]:
        return [p for p, t in self.pairs.items() if t.kind == "cointegrated"]

    def trap_pairs(self) -> list[tuple[str, str]]:
        return [p for p, t in self.pairs.items() if t.kind == "correlated_rw"]

    def break_pairs(self) -> list[tuple[str, str]]:
        return [p for p, t in self.pairs.items() if t.kind == "regime_break"]


def simulate_ou(
    n: int,
    kappa: float,
    sigma: float,
    mu: float = 0.0,
    x0: Optional[float] = None,
    dt: float = 1.0,
    seed: SeedLike = None,
) -> np.ndarray:
    """Simulate an Ornstein-Uhlenbeck path with the *exact* discretisation.

    dX = kappa (mu - X) dt + sigma dW, sampled at spacing ``dt`` via
    X_{t+1} = mu + (X_t - mu) e^{-kappa dt} + eps,
    eps ~ N(0, sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa)).

    Parameters
    ----------
    n : int
        Number of observations (including the initial point).
    kappa : float
        Mean-reversion speed per unit of ``dt`` (per day when dt=1). Must be
        >= 0; kappa == 0 degenerates to a random walk with per-step std
        ``sigma * sqrt(dt)``.
    sigma : float
        Diffusion volatility per sqrt(time unit).
    mu : float
        Long-run mean.
    x0 : float, optional
        Initial value; defaults to a draw from the stationary distribution
        (or ``mu`` when kappa == 0).
    dt : float
        Sampling interval in the same time unit as kappa (default 1 day).
    seed : int | Generator | None
        Randomness control.

    Returns
    -------
    numpy.ndarray of shape (n,)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    rng = make_rng(seed)
    x = np.empty(n)
    if kappa == 0.0:
        step_sd = sigma * np.sqrt(dt)
        x[0] = mu if x0 is None else x0
        x[1:] = x[0] + np.cumsum(rng.normal(0.0, step_sd, size=n - 1))
        return x
    phi = np.exp(-kappa * dt)
    stat_sd = sigma / np.sqrt(2.0 * kappa)
    step_sd = stat_sd * np.sqrt(1.0 - phi**2)
    if x0 is None:
        x[0] = mu + stat_sd * rng.standard_normal()
    else:
        x[0] = x0
    eps = rng.normal(0.0, step_sd, size=n - 1)
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + eps[t - 1]
    return x


def _random_walk(
    n: int, rng: np.random.Generator, p0: float = 100.0, step_sd: float = 1.0
) -> np.ndarray:
    """Arithmetic random walk floored at ``_PRICE_FLOOR``."""
    path = p0 + np.concatenate([[0.0], np.cumsum(rng.normal(0.0, step_sd, n - 1))])
    return np.maximum(path, _PRICE_FLOOR)


def cointegrated_pair(
    n: int = 1500,
    beta: float = 1.5,
    alpha: float = 0.0,
    kappa: float = 0.05,
    sigma: float = 1.0,
    mu: float = 0.0,
    p0: float = 100.0,
    step_sd: float = 1.0,
    seed: SeedLike = None,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, PairTruth]:
    """Generate a truly cointegrated pair Y_t = alpha + beta * X_t + s_t.

    X is an arithmetic random walk (the common stochastic trend) and s is a
    stationary OU spread with known (kappa, sigma, mu). Engle-Granger run as
    Y on X should recover ``beta`` and reject a unit root in the residuals.

    Returns
    -------
    (prices, truth) : (pandas.DataFrame with columns ["Y", "X"], PairTruth)
    """
    rng = make_rng(seed)
    x = _random_walk(n, rng, p0=p0, step_sd=step_sd)
    s = simulate_ou(n, kappa=kappa, sigma=sigma, mu=mu, dt=1.0, seed=rng)
    y = np.maximum(alpha + beta * x + s, _PRICE_FLOOR)
    df = pd.DataFrame({"Y": y, "X": x}, index=business_index(n, start))
    truth = PairTruth(
        kind="cointegrated",
        beta=beta,
        alpha=alpha,
        kappa=kappa,
        sigma=sigma,
        mu=mu,
        half_life=np.log(2.0) / kappa if kappa > 0 else np.inf,
    )
    return df, truth


def correlated_random_walks(
    n: int = 1500,
    rho: float = 0.9,
    p0: float = 100.0,
    step_sd: float = 0.7,
    seed: SeedLike = None,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, PairTruth]:
    """The classic trap: correlated increments, independent levels.

    Two random walks whose daily *increments* have correlation ``rho``.
    Return correlation is high by construction, but each level series has its
    own unit root and there is no linear combination that is stationary:
    the pair is NOT cointegrated and the spread drifts without bound.

    Returns
    -------
    (prices, truth) : (DataFrame with columns ["A", "B"], PairTruth)
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    rng = make_rng(seed)
    z = rng.standard_normal((2, n - 1))
    e1 = z[0]
    e2 = rho * z[0] + np.sqrt(1.0 - rho**2) * z[1]
    a = np.maximum(p0 + np.concatenate([[0.0], np.cumsum(step_sd * e1)]), _PRICE_FLOOR)
    b = np.maximum(p0 + np.concatenate([[0.0], np.cumsum(step_sd * e2)]), _PRICE_FLOOR)
    df = pd.DataFrame({"A": a, "B": b}, index=business_index(n, start))
    return df, PairTruth(kind="correlated_rw")


def regime_break_pair(
    n: int = 1500,
    break_frac: float = 0.5,
    beta: float = 1.2,
    kappa: float = 0.06,
    sigma: float = 0.8,
    drift_after: float = 0.15,
    p0: float = 100.0,
    step_sd: float = 1.0,
    seed: SeedLike = None,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, PairTruth]:
    """Pair whose cointegration dies mid-sample.

    Before the break the spread is OU(kappa, sigma). From the break onwards
    the spread becomes a random walk with drift ``drift_after`` dollars/day —
    mean reversion never resumes, so convergence trades opened after the
    break lose money on average.

    Returns
    -------
    (prices, truth) : (DataFrame with columns ["Y", "X"], PairTruth) where
    ``truth.break_index`` is the first index of the broken regime.
    """
    if not 0.0 < break_frac < 1.0:
        raise ValueError(f"break_frac must be in (0, 1), got {break_frac}")
    rng = make_rng(seed)
    k = int(n * break_frac)
    x = _random_walk(n, rng, p0=p0, step_sd=step_sd)
    s = np.empty(n)
    s[:k] = simulate_ou(k, kappa=kappa, sigma=sigma, mu=0.0, dt=1.0, seed=rng)
    # post-break: unit-root spread with drift, continuous at the break point
    inc = drift_after + rng.normal(0.0, sigma, size=n - k)
    s[k:] = s[k - 1] + np.cumsum(inc)
    y = np.maximum(beta * x + s, _PRICE_FLOOR)
    df = pd.DataFrame({"Y": y, "X": x}, index=business_index(n, start))
    truth = PairTruth(
        kind="regime_break",
        beta=beta,
        kappa=kappa,
        sigma=sigma,
        half_life=np.log(2.0) / kappa,
        break_index=k,
    )
    return df, truth


def mixed_panel(
    n: int = 1500,
    n_cointegrated: int = 4,
    n_trap: int = 3,
    n_break: int = 1,
    n_idiosyncratic: int = 4,
    seed: SeedLike = 7,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, PanelTruth]:
    """Sector-tagged panel mixing all regimes plus idiosyncratic names.

    Layout: each cointegrated / trap / break pair contributes two tickers in
    the same synthetic sector; idiosyncratic names are independent random
    walks spread across sectors. Ticker naming encodes the truth for easy
    debugging (CO_*, TRAP_*, BRK_*, IDIO_*), but the selection code never
    reads the names — only prices and sector tags.

    Returns
    -------
    (prices, truth) : (DataFrame of prices, PanelTruth with sector map and
    per-pair ground truth).
    """
    rng = make_rng(seed)
    sectors = ["TECH", "ENERGY", "FINS", "HEALTH"]
    frames: list[pd.Series] = []
    truth = PanelTruth()

    def sector_for(i: int) -> str:
        return sectors[i % len(sectors)]

    for i in range(n_cointegrated):
        beta = float(rng.uniform(0.8, 1.8))
        kappa = float(rng.uniform(0.05, 0.15))
        sigma = float(rng.uniform(0.6, 1.2))
        df, t = cointegrated_pair(
            n=n, beta=beta, kappa=kappa, sigma=sigma, seed=rng, start=start
        )
        a, b = f"CO{i}_Y", f"CO{i}_X"
        frames += [df["Y"].rename(a), df["X"].rename(b)]
        sec = sector_for(i)
        truth.sectors[a] = sec
        truth.sectors[b] = sec
        truth.pairs[(a, b)] = t

    for i in range(n_trap):
        df, t = correlated_random_walks(n=n, rho=0.92, seed=rng, start=start)
        a, b = f"TRAP{i}_A", f"TRAP{i}_B"
        frames += [df["A"].rename(a), df["B"].rename(b)]
        sec = sector_for(n_cointegrated + i)
        truth.sectors[a] = sec
        truth.sectors[b] = sec
        truth.pairs[(a, b)] = t

    for i in range(n_break):
        df, t = regime_break_pair(n=n, seed=rng, start=start)
        a, b = f"BRK{i}_Y", f"BRK{i}_X"
        frames += [df["Y"].rename(a), df["X"].rename(b)]
        sec = sector_for(n_cointegrated + n_trap + i)
        truth.sectors[a] = sec
        truth.sectors[b] = sec
        truth.pairs[(a, b)] = t

    for i in range(n_idiosyncratic):
        name = f"IDIO{i}"
        path = _random_walk(n, rng, p0=float(rng.uniform(50, 150)), step_sd=1.0)
        frames.append(pd.Series(path, index=business_index(n, start), name=name))
        truth.sectors[name] = sector_for(i)

    prices = pd.concat(frames, axis=1)
    return prices, truth
