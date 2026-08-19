r"""Counterparty pre-settlement risk on FX forwards: EE/PFE profiles, netting, CVA.

Model
-----
Spot follows Garman-Kohlhagen GBM under the domestic (quote-currency) measure:

.. math:: dS_t = (r_d - r_f) S_t\,dt + \sigma S_t\,dW_t

with pairs quoted BASE/QUOTE (EURUSD = USD per EUR), domestic rate = quote
rate ``r_d``, foreign = base rate ``r_f`` (portfolio FX conventions).

The MTM at time t of a forward to *buy* base notional N at strike K maturing
at T is ``V_t = N (F_t - K) e^{-r_d (T-t)}`` with ``F_t = S_t e^{(r_d-r_f)(T-t)}``.
Current exposure is ``max(V_t, 0)``; EE(t) is its mean over paths and
PFE_q(t) its q-quantile.

Exposure shape for a single forward: there are **no intermediate cashflows**,
so uncertainty accumulates all the way to maturity and the exposure profile
*grows to maturity* like ``sqrt(t)`` (concave, monotone increasing) — unlike
an interest-rate swap, whose amortising remaining cashflows create the famous
mid-life hump.  Both the monotone growth and the concavity are unit-tested
empirically on seeded paths.

CVA (unilateral, no wrong-way risk):

.. math:: \mathrm{CVA} = LGD \sum_i EE(t_i)\,[PD(t_i)-PD(t_{i-1})]\,e^{-r_d t_i}

with the PD term structure built from the scorecard's 1-year PD via a **flat
hazard**: ``h = -ln(1 - PD_{1y})``, ``PD(t) = 1 - e^{-ht}`` — a documented
simplification (real curves are humped for low-grade sovereigns).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "FXForward",
    "ExposureProfile",
    "simulate_fx_paths",
    "forward_mtm",
    "exposure_profile",
    "netting_set_profile",
    "hazard_from_pd1y",
    "pd_term_structure",
    "cva",
    "cva_for_forward",
]


@dataclass(frozen=True)
class FXForward:
    """FX forward: buy (+) or sell (-) ``notional_base`` of BASE at strike K.

    Attributes
    ----------
    pair : str
        BASE/QUOTE pair, e.g. "EURUSD".
    notional_base : float
        Absolute base-currency notional (>= 0).
    strike : float
        Delivery rate K (quote per base).
    maturity : float
        Maturity T in years (ACT/365-style year fractions).
    buy_base : bool
        True = long the base currency forward.
    """

    pair: str
    notional_base: float
    strike: float
    maturity: float
    buy_base: bool = True

    def __post_init__(self) -> None:
        if not (np.isfinite(self.notional_base) and self.notional_base >= 0):
            raise ValueError("notional_base must be finite and >= 0")
        if not (np.isfinite(self.strike) and self.strike > 0):
            raise ValueError("strike must be finite and > 0")
        if not np.isfinite(self.maturity):
            raise ValueError("maturity must be finite")


@dataclass(frozen=True)
class ExposureProfile:
    """Simulated exposure profile on a fixed time grid."""

    times: np.ndarray            # (m,) year fractions > 0
    ee: np.ndarray               # expected exposure E[max(V,0)]
    pfe: dict[float, np.ndarray]  # quantile -> PFE path, e.g. {0.95: ..., 0.99: ...}

    def peak_pfe(self, q: float) -> float:
        return float(np.max(self.pfe[q])) if self.pfe[q].size else 0.0


def simulate_fx_paths(
    spot: float,
    vol: float,
    r_d: float,
    r_f: float,
    times: np.ndarray,
    n_paths: int,
    seed: int = 0,
) -> np.ndarray:
    """Exact-scheme GBM spot paths on a time grid (no discretisation error).

    Parameters
    ----------
    spot : float
        S_0, quote units per base unit (> 0).
    vol : float
        Lognormal volatility, annualised.
    r_d, r_f : float
        Domestic (quote) and foreign (base) continuously-compounded rates.
    times : ndarray
        Strictly increasing positive year fractions, shape (m,).
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Generator seed.

    Returns
    -------
    ndarray, shape (n_paths, m)
    """
    if not (np.isfinite(spot) and np.isfinite(vol) and np.isfinite(r_d) and np.isfinite(r_f)):
        raise ValueError("spot, vol, r_d, r_f must all be finite")
    if spot <= 0 or vol < 0:
        raise ValueError("spot must be > 0 and vol >= 0")
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError("times must be finite, strictly increasing and positive")
    rng = np.random.default_rng(seed)
    dt = np.diff(np.r_[0.0, t])
    z = rng.standard_normal((n_paths, t.size))
    incr = (r_d - r_f - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    return spot * np.exp(np.cumsum(incr, axis=1))


def forward_mtm(
    fwd: FXForward,
    spot: np.ndarray | float,
    t: float,
    r_d: float,
    r_f: float,
) -> np.ndarray | float:
    """MTM (quote currency) of the forward at time t given spot level(s).

    ``V = sign * N * (F_t - K) * e^{-r_d (T - t)}``; at t == T this is the
    settlement payoff ``sign * N * (S_T - K)``; 0 if matured (t > T).
    """
    tau = fwd.maturity - t
    if tau < 0:
        return np.zeros_like(np.asarray(spot, dtype=float)) if not np.isscalar(spot) else 0.0
    f_t = np.asarray(spot, dtype=float) * np.exp((r_d - r_f) * tau)
    sign = 1.0 if fwd.buy_base else -1.0
    v = sign * fwd.notional_base * (f_t - fwd.strike) * np.exp(-r_d * tau)
    return float(v) if np.isscalar(spot) else v


def _grid(maturity: float, n_steps: int) -> np.ndarray:
    return np.linspace(maturity / n_steps, maturity, n_steps)


def exposure_profile(
    fwd: FXForward,
    spot: float,
    vol: float,
    r_d: float,
    r_f: float,
    n_steps: int = 24,
    n_paths: int = 50_000,
    quantiles: tuple[float, ...] = (0.95, 0.99),
    seed: int = 0,
) -> ExposureProfile:
    """EE and PFE profile of a single FX forward via seeded GBM simulation.

    Exposure at each grid time is ``max(V_t, 0)`` per path.  A matured or
    zero-notional forward returns an all-zero profile on the (possibly empty)
    grid.
    """
    if fwd.maturity <= 0 or fwd.notional_base == 0:
        times = _grid(max(fwd.maturity, 0.0), n_steps) if fwd.maturity > 0 else np.array([])
        z = np.zeros_like(times)
        return ExposureProfile(times, z, {q: z.copy() for q in quantiles})
    times = _grid(fwd.maturity, n_steps)
    paths = simulate_fx_paths(spot, vol, r_d, r_f, times, n_paths, seed)
    ee = np.empty(times.size)
    pfe = {q: np.empty(times.size) for q in quantiles}
    for j, t in enumerate(times):
        v = forward_mtm(fwd, paths[:, j], t, r_d, r_f)
        e = np.maximum(v, 0.0)
        ee[j] = e.mean()
        for q in quantiles:
            pfe[q][j] = np.quantile(e, q)
    return ExposureProfile(times, ee, pfe)


def netting_set_profile(
    forwards: list[FXForward],
    spot: float,
    vol: float,
    r_d: float,
    r_f: float,
    netting: bool = True,
    n_steps: int = 24,
    n_paths: int = 50_000,
    quantiles: tuple[float, ...] = (0.95, 0.99),
    seed: int = 0,
) -> ExposureProfile:
    """Exposure profile of a set of same-pair forwards with/without netting.

    With a netting agreement the exposure is ``max(sum_i V_i, 0)``; without,
    each trade defaults severally: ``sum_i max(V_i, 0)``.  Netting therefore
    always weakly reduces exposure path-by-path, with equality iff all trade
    MTMs share the same sign on every path (e.g. identical same-direction
    trades) — both bounds unit-tested.  All trades share the driving spot
    paths (same pair, same seed): the comparison is apples-to-apples.
    """
    if not forwards:
        raise ValueError("empty netting set")
    pairs = {f.pair for f in forwards}
    if len(pairs) > 1:
        raise ValueError(f"netting-set simulator supports one pair, got {pairs}")
    horizon = max(f.maturity for f in forwards)
    if horizon <= 0:
        z = np.array([])
        return ExposureProfile(z, z.copy(), {q: z.copy() for q in quantiles})
    times = _grid(horizon, n_steps)
    paths = simulate_fx_paths(spot, vol, r_d, r_f, times, n_paths, seed)
    ee = np.empty(times.size)
    pfe = {q: np.empty(times.size) for q in quantiles}
    for j, t in enumerate(times):
        vs = [forward_mtm(f, paths[:, j], t, r_d, r_f) for f in forwards]
        vs = [np.broadcast_to(v, (n_paths,)) for v in vs]
        if netting:
            e = np.maximum(np.sum(vs, axis=0), 0.0)
        else:
            e = np.sum([np.maximum(v, 0.0) for v in vs], axis=0)
        ee[j] = e.mean()
        for q in quantiles:
            pfe[q][j] = np.quantile(e, q)
    return ExposureProfile(times, ee, pfe)


# ---------------------------------------------------------------------------
# PD term structure from the scorecard 1y PD, and CVA
# ---------------------------------------------------------------------------

def hazard_from_pd1y(pd_1y: float) -> float:
    """Flat hazard rate implied by a 1-year PD: ``h = -ln(1 - PD_1y)``."""
    if not 0.0 <= pd_1y < 1.0:
        raise ValueError("pd_1y must be in [0, 1)")
    return float(-np.log1p(-pd_1y))


def pd_term_structure(pd_1y: float, times: np.ndarray) -> np.ndarray:
    """Cumulative PD at each time under a flat hazard: ``1 - e^{-h t}``.

    Documented simplification: a single 1y PD cannot identify the term
    structure; real sovereign curves are upward-humped for low grades and
    the flat-hazard curve understates front-loaded risk after a downgrade.
    """
    h = hazard_from_pd1y(pd_1y)
    t = np.asarray(times, dtype=float)
    return 1.0 - np.exp(-h * t)


def cva(
    times: np.ndarray,
    ee: np.ndarray,
    cum_pd: np.ndarray,
    lgd: float,
    r_d: float = 0.0,
) -> float:
    """Unilateral CVA: ``LGD * sum_i EE(t_i) dPD_i e^{-r_d t_i}``.

    Parameters
    ----------
    times : ndarray
        Increasing positive grid times t_i (years).
    ee : ndarray
        Expected exposure at each t_i (quote currency).
    cum_pd : ndarray
        Cumulative default probability at each t_i (nondecreasing, in [0,1)).
    lgd : float
        Loss given default fraction in [0, 1].
    r_d : float
        Flat discount rate (continuous compounding).

    Notes
    -----
    Uses the end-of-interval EE (defaults within (t_{i-1}, t_i] valued at
    t_i) — a conservative, standard discretisation.  Independence of
    exposure and default is assumed (no wrong-way risk); VALIDATION.md
    quantifies how badly that understates CVA for an EM sovereign whose
    default coincides with its currency crashing.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(ee, dtype=float)
    q = np.asarray(cum_pd, dtype=float)
    if not (t.shape == e.shape == q.shape):
        raise ValueError("times, ee, cum_pd must share shape")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(e)) and np.all(np.isfinite(q))):
        raise ValueError("times, ee, cum_pd must be finite (no NaN/Inf)")
    if np.any(e < 0):
        raise ValueError("ee must be >= 0 (expected exposure is a positive part)")
    if np.any((q < 0) | (q >= 1)):
        raise ValueError("cum_pd must be in [0, 1)")
    if not 0.0 <= lgd <= 1.0:
        raise ValueError("lgd must be in [0,1]")
    if np.any(np.diff(q) < -1e-12):
        raise ValueError("cum_pd must be nondecreasing")
    if t.size == 0:
        return 0.0
    dpd = np.diff(np.r_[0.0, q])
    disc = np.exp(-r_d * t)
    return float(lgd * np.sum(e * dpd * disc))


def cva_for_forward(
    fwd: FXForward,
    spot: float,
    vol: float,
    r_d: float,
    r_f: float,
    pd_1y: float,
    lgd: float,
    n_steps: int = 24,
    n_paths: int = 50_000,
    seed: int = 0,
) -> tuple[float, ExposureProfile]:
    """CVA of a single forward against a counterparty with given 1y PD.

    Returns (cva_value, exposure_profile).  ``pd_1y = 0`` gives exactly 0.
    """
    prof = exposure_profile(fwd, spot, vol, r_d, r_f, n_steps, n_paths, seed=seed)
    if prof.times.size == 0:
        return 0.0, prof
    q = pd_term_structure(pd_1y, prof.times)
    return cva(prof.times, prof.ee, q, lgd, r_d), prof
