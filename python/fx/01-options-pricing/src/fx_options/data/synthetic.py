"""Deterministic synthetic FX data: GBM paths and stylised vol quotes.

Everything here is seeded and offline — this is the only data source the
test suite touches (see CONVENTIONS.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .._common import validate_inputs

__all__ = ["gbm_fx_paths", "VolQuote", "synthetic_vol_quotes"]


def gbm_fx_paths(S0: float, T: float, r_d: float, r_f: float, sigma: float,
                 n_steps: int, n_paths: int,
                 rng: np.random.Generator | int | None = 0,
                 mu: float | None = None) -> np.ndarray:
    """Simulate GBM FX spot paths.

    Parameters
    ----------
    S0 : float
        Initial spot (domestic per unit foreign), > 0.
    T : float
        Horizon in years, > 0.
    r_d, r_f : float
        Domestic / foreign continuously compounded rates.
    sigma : float
        Annualised volatility, >= 0.
    n_steps, n_paths : int
        Time steps and number of paths, >= 1.
    rng : numpy.random.Generator or int or None
        Explicit generator or seed (deterministic default seed 0).
    mu : float, optional
        Drift; defaults to the domestic risk-neutral drift ``r_d - r_f``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_paths, n_steps + 1)``; column 0 is ``S0``.
    """
    validate_inputs(S0, S0, T, r_d, r_f, sigma)
    if T <= 0.0:
        raise ValueError("gbm_fx_paths requires T > 0")
    for name, n in (("n_steps", n_steps), ("n_paths", n_paths)):
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"{name} must be a positive integer, got {n!r}")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    drift = (r_d - r_f) if mu is None else mu
    dt = T / n_steps
    z = gen.standard_normal((n_paths, n_steps))
    increments = (drift - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    return S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


@dataclass(frozen=True)
class VolQuote:
    """One tenor of market-style FX vol quotes (all vols annualised).

    ``atm`` is the delta-neutral-straddle vol; ``rr25``/``rr10`` are risk
    reversals (call vol minus put vol at 25/10 delta — skew direction);
    ``bf25``/``bf10`` are butterflies (average wing vol minus ATM —
    smile curvature).  Smile *construction* from these quotes is
    project 9; here they only feed documentation/examples.
    """

    tenor_years: float
    atm: float
    rr25: float
    bf25: float
    rr10: float
    bf10: float


def synthetic_vol_quotes(base_atm: float = 0.10, skew: float = -0.01,
                         smile: float = 0.003,
                         tenors: tuple[float, ...] = (1 / 12, 0.25, 0.5, 1.0),
                         rng: np.random.Generator | int | None = 0,
                         noise: float = 0.0005) -> list[VolQuote]:
    """Generate a stylised, seeded set of ATM/RR/BF quotes across tenors.

    ATM follows a mildly upward term structure; RR keeps the sign of
    ``skew`` (negative = puts on the base ccy bid, typical for EURUSD
    risk-off skew); BF widens with tenor.  Small seeded noise mimics
    market scatter while remaining fully deterministic.
    """
    for name, value in (("base_atm", base_atm), ("skew", skew),
                        ("smile", smile), ("noise", noise)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if base_atm <= 0:
        raise ValueError(f"base_atm must be positive, got {base_atm}")
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    quotes = []
    for t in tenors:
        if not math.isfinite(t) or t <= 0:
            raise ValueError(f"tenors must be positive and finite, got {t!r}")
        eps = gen.normal(0.0, noise, size=5)
        term = 1.0 + 0.15 * math.log1p(t)
        quotes.append(VolQuote(
            tenor_years=t,
            atm=base_atm * term + eps[0],
            rr25=skew * math.sqrt(t / 0.25) + eps[1],
            bf25=smile * term + abs(eps[2]),
            rr10=1.8 * skew * math.sqrt(t / 0.25) + eps[3],
            bf10=3.2 * smile * term + abs(eps[4]),
        ))
    return quotes
