"""Deterministic synthetic market data for examples and tests (offline).

Everything here is seeded and reproducible — the test suite never touches
the network. Conventions: annualised continuously compounded rates, ACT/365F
year fractions, annualised log-return vols.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..black_scholes import bs_price, forward_price, validate_inputs

__all__ = ["gbm_paths", "synthetic_chain", "skew_vol"]


def gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | np.random.Generator | None = 0,
) -> np.ndarray:
    """Simulate exact-scheme GBM paths on an equally spaced grid.

    ``S_{t+dt} = S_t exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)``.

    Parameters
    ----------
    S0 : float
        Initial spot, > 0.
    mu : float
        Annualised continuously compounded drift.
    sigma : float
        Annualised volatility, >= 0.
    T : float
        Horizon in years (ACT/365F), > 0.
    n_steps : int
        Number of time steps (grid has ``n_steps + 1`` points), >= 1.
    n_paths : int
        Number of paths, >= 1.
    seed : int or numpy.random.Generator or None
        Explicit seed; same seed => identical paths.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_paths, n_steps + 1)`` with ``paths[:, 0] == S0``.

    Raises
    ------
    ValueError
        On non-positive ``S0``/``T`` or invalid sizes.
    """
    validate_inputs(S0, 1.0, T, sigma)
    if S0 <= 0 or T <= 0:
        raise ValueError("gbm_paths requires S0 > 0 and T > 0")
    if n_steps < 1 or n_paths < 1:
        raise ValueError("n_steps and n_paths must be >= 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)


def skew_vol(
    K: float | np.ndarray,
    F: float,
    T: float,
    base_vol: float = 0.20,
    skew: float = -0.10,
    smile: float = 0.05,
    vol_floor: float = 0.05,
) -> float | np.ndarray:
    """Mild parametric equity skew: quadratic in log-moneyness.

    ``iv(K) = base_vol + skew * m + smile * m^2`` with
    ``m = ln(K / F) / sqrt(T)``, floored at ``vol_floor``. The negative
    default ``skew`` gives the classic equity pattern (puts rich).

    Parameters
    ----------
    K : float or numpy.ndarray
        Strike(s), > 0.
    F : float
        Forward for the expiry, > 0.
    T : float
        Expiry in years, > 0.
    base_vol, skew, smile, vol_floor : float
        Parameters of the quadratic; ``vol_floor`` keeps vols positive.

    Returns
    -------
    float or numpy.ndarray
        Annualised implied vol(s), >= ``vol_floor``.
    """
    m = np.log(np.asarray(K, dtype=float) / F) / math.sqrt(T)
    iv = base_vol + skew * m + smile * m * m
    out = np.maximum(iv, vol_floor)
    return float(out) if np.isscalar(K) else out


def synthetic_chain(
    S0: float = 100.0,
    r: float = 0.03,
    q: float = 0.01,
    expiries: tuple[float, ...] = (0.083, 0.25, 0.5, 1.0),
    n_strikes: int = 11,
    strike_width: float = 0.30,
    base_vol: float = 0.20,
    skew: float = -0.10,
    smile: float = 0.05,
) -> pd.DataFrame:
    """Generate a synthetic option chain with a mild equity skew.

    Strikes span ``[S0 (1 - strike_width), S0 (1 + strike_width)]`` per
    expiry; each (strike, expiry) gets a call and a put priced with
    Black-Scholes at the skewed vol, so implied vols round-trip exactly.

    Parameters
    ----------
    S0 : float
        Spot, > 0.
    r, q : float
        Annualised continuously compounded rate and dividend yield.
    expiries : tuple of float
        Expiries in years (ACT/365F), each > 0.
    n_strikes : int
        Strikes per expiry, >= 2.
    strike_width : float
        Half-width of the strike range as a fraction of spot.
    base_vol, skew, smile : float
        Parameters passed to :func:`skew_vol`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``expiry`` (years), ``strike``, ``type`` ('call'/'put'),
        ``iv`` (annualised), ``price`` (currency units), ``forward``.

    Raises
    ------
    ValueError
        On invalid inputs.
    """
    if S0 <= 0:
        raise ValueError("S0 must be > 0")
    if n_strikes < 2:
        raise ValueError("n_strikes must be >= 2")
    rows: list[dict[str, object]] = []
    strikes = np.linspace(S0 * (1 - strike_width), S0 * (1 + strike_width), n_strikes)
    for T in expiries:
        if T <= 0:
            raise ValueError("expiries must all be > 0")
        F = forward_price(S0, T, r, q)
        for K in strikes:
            iv = skew_vol(float(K), F, T, base_vol, skew, smile)
            for opt in ("call", "put"):
                rows.append({
                    "expiry": T,
                    "strike": float(K),
                    "type": opt,
                    "iv": iv,
                    "price": bs_price(S0, float(K), T, r, iv, q, opt),
                    "forward": F,
                })
    return pd.DataFrame(rows)
