"""Seeded synthetic FX option markets.

Three stylised presets plus a Heston ground-truth mode:

* :func:`eurusd_market` - G10 major: moderate vols, small negative RR
  (mild EUR-put skew), modest BF, *unadjusted* deltas (spot to 1y,
  forward beyond - the interbank convention for EURUSD).
* :func:`usdjpy_market` - JPY-style pair: large negative RR (persistent
  USD-put/JPY-call skew), premium-adjusted deltas and the pa DNS ATM
  (premium paid in USD, the base currency).
* :func:`em_high_vol_market` - stressed EM pair: 35% ATM, large
  *positive* RR (devaluation skew on the topside), fat BF.
* :func:`market_from_heston` - quotes generated from known Heston
  parameters (smile -> pillar strikes by fixed point -> ATM/RR/BF),
  used for the calibration recovery tests.

All numbers are stylised but shaped from realistic 2024-era levels.
Optional quote noise is Gaussian, seeded, in vol points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from ..calibration import CalibrationSlice
from ..garman_kohlhagen import gk_forward, implied_vol
from ..heston import HestonParams, price_cos
from ..smile_from_quotes import (
    PILLAR_LABELS,
    SmileQuotes,
    atm_dns_strike,
    quotes_from_vols,
    solve_pillar_strikes,
    strike_from_delta,
    vols_from_quotes,
)

__all__ = [
    "MarketSlice",
    "FXMarketData",
    "eurusd_market",
    "usdjpy_market",
    "em_high_vol_market",
    "market_from_heston",
    "calibration_slices",
]

TENORS: tuple[tuple[str, float], ...] = (
    ("1w", 7.0 / 365.0),
    ("1m", 30.0 / 365.0),
    ("3m", 91.0 / 365.0),
    ("6m", 182.0 / 365.0),
    ("1y", 1.0),
    ("2y", 2.0),
)


@dataclass(frozen=True)
class MarketSlice:
    """One expiry of raw broker quotes."""

    label: str
    T: float
    r_d: float
    r_f: float
    quotes: SmileQuotes
    convention: str  # native delta convention for these quotes


@dataclass(frozen=True)
class FXMarketData:
    """A full synthetic market: spot plus quote slices per tenor."""

    pair: str
    S: float
    slices: tuple[MarketSlice, ...]

    @property
    def tenor_labels(self) -> list[str]:
        return [s.label for s in self.slices]


def _apply_noise(
    quotes: SmileQuotes, noise: float, rng: np.random.Generator
) -> SmileQuotes:
    if noise <= 0.0:
        return quotes
    e = rng.normal(0.0, noise, size=5)
    return SmileQuotes(
        atm=quotes.atm + e[0],
        rr25=quotes.rr25 + e[1],
        bf25=max(quotes.bf25 + e[2], 0.0),
        rr10=quotes.rr10 + e[3],
        bf10=max(quotes.bf10 + e[4], 0.0),
    )


def _build(
    pair: str,
    S: float,
    r_d: Sequence[float],
    r_f: Sequence[float],
    atm: Sequence[float],
    rr25: Sequence[float],
    bf25: Sequence[float],
    rr10: Sequence[float],
    bf10: Sequence[float],
    conventions: Sequence[str],
    noise: float,
    seed: int,
) -> FXMarketData:
    rng = np.random.default_rng(seed)
    slices = []
    for i, (label, T) in enumerate(TENORS):
        q = SmileQuotes(atm[i], rr25[i], bf25[i], rr10[i], bf10[i])
        q = _apply_noise(q, noise, rng)
        slices.append(MarketSlice(label, T, r_d[i], r_f[i], q, conventions[i]))
    return FXMarketData(pair, S, tuple(slices))


def eurusd_market(noise: float = 0.0, seed: int = 0) -> FXMarketData:
    """EURUSD-like preset: mild skew, small |RR|, moderate BF.

    Unadjusted deltas (premium in USD, the quote currency): spot delta
    out to 1y, forward delta at 2y - the interbank convention.
    """
    return _build(
        pair="EURUSD",
        S=1.10,
        r_d=[0.0460, 0.0455, 0.0445, 0.0430, 0.0410, 0.0390],  # USD
        r_f=[0.0360, 0.0355, 0.0345, 0.0330, 0.0310, 0.0290],  # EUR
        atm=[0.0780, 0.0745, 0.0720, 0.0715, 0.0730, 0.0750],
        rr25=[-0.0020, -0.0030, -0.0045, -0.0055, -0.0060, -0.0060],
        bf25=[0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0033],
        rr10=[-0.0036, -0.0054, -0.0080, -0.0098, -0.0107, -0.0107],
        bf10=[0.0033, 0.0050, 0.0066, 0.0083, 0.0099, 0.0109],
        conventions=["spot", "spot", "spot", "spot", "spot", "forward"],
        noise=noise,
        seed=seed,
    )


def usdjpy_market(noise: float = 0.0, seed: int = 0) -> FXMarketData:
    """USDJPY-like preset: strong JPY-call (USD-put) skew, pa deltas.

    Premium is paid in USD (the base currency), so deltas are
    premium-adjusted and ATM is the pa delta-neutral straddle
    ``K = F exp(-sigma^2 T / 2)``: spot pa out to 1y, forward pa at 2y.
    """
    return _build(
        pair="USDJPY",
        S=150.0,
        r_d=[0.0010, 0.0015, 0.0020, 0.0030, 0.0040, 0.0050],  # JPY
        r_f=[0.0460, 0.0455, 0.0445, 0.0430, 0.0410, 0.0390],  # USD
        atm=[0.1000, 0.1005, 0.1010, 0.1025, 0.1045, 0.1065],
        rr25=[-0.0060, -0.0110, -0.0170, -0.0215, -0.0245, -0.0255],
        bf25=[0.0015, 0.0022, 0.0030, 0.0037, 0.0043, 0.0047],
        rr10=[-0.0108, -0.0198, -0.0306, -0.0387, -0.0441, -0.0459],
        bf10=[0.0044, 0.0064, 0.0087, 0.0107, 0.0125, 0.0136],
        conventions=["spot_pa", "spot_pa", "spot_pa", "spot_pa", "spot_pa", "forward_pa"],
        noise=noise,
        seed=seed,
    )


def em_high_vol_market(noise: float = 0.0, seed: int = 0) -> FXMarketData:
    """Stressed EM preset (USD/EM): 35% ATM, big positive RR
    (devaluation risk prices the topside), fat butterflies.  Unadjusted
    spot deltas throughout for simplicity."""
    return _build(
        pair="USDEM",
        S=8.50,
        r_d=[0.30, 0.30, 0.29, 0.28, 0.27, 0.26],  # EM rate (quote ccy)
        r_f=[0.046, 0.045, 0.044, 0.043, 0.041, 0.039],  # USD
        atm=[0.330, 0.340, 0.350, 0.350, 0.345, 0.340],
        rr25=[0.040, 0.050, 0.060, 0.062, 0.060, 0.058],
        bf25=[0.008, 0.010, 0.012, 0.012, 0.012, 0.011],
        rr10=[0.072, 0.090, 0.108, 0.112, 0.108, 0.104],
        bf10=[0.026, 0.033, 0.040, 0.040, 0.040, 0.036],
        conventions=["spot"] * 6,
        noise=noise,
        seed=seed,
    )


# ----------------------------------------------------------------------
# Ground-truth Heston mode
# ----------------------------------------------------------------------

def _heston_vol_fn(S, T, r_d, r_f, params, N=512):
    def vol(K: float) -> float:
        pr = float(price_cos(S, K, T, r_d, r_f, params, cp=-1, N=N))
        return implied_vol(pr, S, K, T, r_d, r_f, cp=-1)

    return vol


def _heston_pillar_vols(
    S: float, T: float, r_d: float, r_f: float, params: HestonParams,
    convention: str, n_iter: int = 60, tol: float = 1e-12,
) -> tuple[dict[str, float], dict[str, float]]:
    """Solve the five smile-consistent pillar strikes and vols under
    Heston by fixed-point iteration (strike -> vol -> strike ...)."""
    vol_fn = _heston_vol_fn(S, T, r_d, r_f, params)
    F = gk_forward(S, T, r_d, r_f)
    pa = convention.endswith("_pa")

    def solve(delta: float | None, cp: int) -> tuple[float, float]:
        K = F
        sig = vol_fn(K)
        for _ in range(n_iter):
            K_new = (
                atm_dns_strike(F, sig, T, pa)
                if delta is None
                else strike_from_delta(delta, cp, sig, S, T, r_d, r_f, convention)
            )
            sig_new = vol_fn(K_new)
            if abs(sig_new - sig) < tol and abs(K_new - K) < tol * F:
                K, sig = K_new, sig_new
                break
            K, sig = K_new, sig_new
        return K, sig

    specs = {"10p": (0.10, -1), "25p": (0.25, -1), "atm": (None, 1),
             "25c": (0.25, +1), "10c": (0.10, +1)}
    strikes, vols = {}, {}
    for label, (d, cp) in specs.items():
        K, sig = solve(d, cp)
        strikes[label], vols[label] = K, sig
    return strikes, vols


def market_from_heston(
    params: HestonParams,
    pair: str = "EURUSD-GT",
    S: float = 1.10,
    r_d: Sequence[float] | None = None,
    r_f: Sequence[float] | None = None,
    conventions: Union[str, Sequence[str]] = "spot",
    tenors: Sequence[tuple[str, float]] = TENORS,
    noise: float = 0.0,
    seed: int = 0,
) -> FXMarketData:
    """Generate broker quotes from known Heston parameters.

    For each tenor the smile-consistent pillar strikes are solved by
    fixed point on the Heston implied-vol curve, and {ATM, RR, BF}
    quotes are read off the five pillar vols.  Used for the
    calibration ground-truth recovery path.
    """
    if r_d is None:
        r_d = [0.0450, 0.0445, 0.0440, 0.0430, 0.0410, 0.0390][: len(tenors)]
    if r_f is None:
        r_f = [0.0350, 0.0345, 0.0340, 0.0330, 0.0310, 0.0290][: len(tenors)]
    if isinstance(conventions, str):
        conventions = [conventions] * len(tenors)
    rng = np.random.default_rng(seed)
    slices = []
    for i, (label, T) in enumerate(tenors):
        _, vols = _heston_pillar_vols(S, T, r_d[i], r_f[i], params, conventions[i])
        q = _apply_noise(quotes_from_vols(vols), noise, rng)
        slices.append(MarketSlice(label, T, r_d[i], r_f[i], q, conventions[i]))
    return FXMarketData(pair, S, tuple(slices))


def calibration_slices(market: FXMarketData) -> list[CalibrationSlice]:
    """Market quotes -> (strike, vol) calibration targets per expiry,
    using each slice's native delta convention for strike placement."""
    out = []
    for ms in market.slices:
        vols = vols_from_quotes(ms.quotes)
        strikes = solve_pillar_strikes(vols, market.S, ms.T, ms.r_d, ms.r_f, ms.convention)
        out.append(
            CalibrationSlice(
                T=ms.T,
                r_d=ms.r_d,
                r_f=ms.r_f,
                strikes=np.array([strikes[p] for p in PILLAR_LABELS]),
                vols=np.array([vols[p] for p in PILLAR_LABELS]),
            )
        )
    return out
