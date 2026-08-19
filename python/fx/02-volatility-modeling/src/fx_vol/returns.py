"""FX return construction, pair inversion and cross-pair triangulation.

Conventions (see CONVENTIONS.md at the portfolio root):

* FX pairs are quoted BASE/QUOTE and written as 6-character strings, e.g.
  ``"EURUSD"`` = number of USD per 1 EUR.
* Returns are **log returns** ``r_t = ln(S_t / S_{t-1})``, unitless per period
  (daily unless stated otherwise). Annualization conventions live in
  :mod:`fx_vol.historical`.
* Inverting a pair maps ``S -> 1/S`` and therefore flips the *sign* of every
  log return, leaving all even moments -- in particular volatility --
  invariant. This is an FX-specific identity with no equity analogue and is
  unit-tested (``tests/test_returns.py``).
* Triangulation: for a cross built from two legs sharing a common currency,
  ``ln S_cross = c1 * ln S_1 + c2 * ln S_2`` with signs ``c1, c2 in {+1, -1}``
  determined by quote direction, hence

      sigma_cross^2 = sigma_1^2 + sigma_2^2 + 2 * c1 * c2 * rho * sigma_1 * sigma_2

  where ``rho`` is the correlation of the two legs' log returns *as quoted*.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "log_returns",
    "invert_prices",
    "invert_returns",
    "pair_currencies",
    "cross_pair_signs",
    "triangulate_prices",
    "triangulate_returns",
    "cross_volatility",
]


def _as_1d_array(values: Sequence[float] | np.ndarray | pd.Series, name: str) -> np.ndarray:
    """Coerce to a 1-D float array, enforcing the NaN policy (reject)."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(
            f"{name} contains NaN or infinite values; fx_vol rejects rather than "
            "silently imputes -- clean or forward-fill the series explicitly first"
        )
    return arr


def _validate_prices(prices: Sequence[float] | np.ndarray | pd.Series, name: str = "prices") -> np.ndarray:
    arr = _as_1d_array(prices, name)
    if arr.size < 2:
        raise ValueError(f"{name} needs at least 2 observations, got {arr.size}")
    if (arr <= 0.0).any():
        raise ValueError(f"{name} must be strictly positive (FX rates), found min={arr.min()}")
    return arr


def log_returns(prices: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Compute log returns ``r_t = ln(S_t / S_{t-1})`` from a price series.

    Parameters
    ----------
    prices : array-like or pandas.Series
        Strictly positive BASE/QUOTE FX rates. NaNs raise ``ValueError``.

    Returns
    -------
    numpy.ndarray or pandas.Series
        Length ``len(prices) - 1``. If a Series is passed, the index of the
        returned Series is ``prices.index[1:]`` (return stamped at the *end*
        of the interval).
    """
    arr = _validate_prices(prices)
    rets = np.diff(np.log(arr))
    if isinstance(prices, pd.Series):
        return pd.Series(rets, index=prices.index[1:], name=prices.name)
    return rets


def invert_prices(prices: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Invert a pair: BASE/QUOTE -> QUOTE/BASE, i.e. ``S -> 1/S``.

    Log returns of the inverted pair are the negated originals, so any
    volatility estimator based on log returns is invariant under inversion.
    """
    arr = _validate_prices(prices)
    inv = 1.0 / arr
    if isinstance(prices, pd.Series):
        return pd.Series(inv, index=prices.index, name=prices.name)
    return inv


def invert_returns(returns: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Map log returns of BASE/QUOTE to log returns of QUOTE/BASE (sign flip)."""
    arr = _as_1d_array(returns, "returns")
    out = -arr
    if isinstance(returns, pd.Series):
        return pd.Series(out, index=returns.index, name=returns.name)
    return out


def pair_currencies(pair: str) -> tuple[str, str]:
    """Split a 6-character pair string into (base, quote), e.g. EURUSD -> (EUR, USD)."""
    if not isinstance(pair, str) or len(pair) != 6 or not pair.isalpha():
        raise ValueError(f"pair must be a 6-letter string like 'EURUSD', got {pair!r}")
    base, quote = pair[:3].upper(), pair[3:].upper()
    if base == quote:
        raise ValueError(f"pair {pair!r} has identical base and quote currency")
    return base, quote


def cross_pair_signs(pair1: str, pair2: str, cross: str) -> tuple[int, int]:
    """Solve ``ln S_cross = c1 ln S_1 + c2 ln S_2`` for signs ``c1, c2``.

    Examples
    --------
    EURJPY from EURUSD and USDJPY  -> (+1, +1)
    EURJPY from EURUSD and JPYUSD  -> (+1, -1)
    CHFJPY from USDCHF and USDJPY  -> (-1, +1)

    Raises
    ------
    ValueError
        If no sign combination reproduces the cross (legs do not share the
        required common currency).
    """
    def vec(pair: str) -> dict[str, int]:
        base, quote = pair_currencies(pair)
        return {base: 1, quote: -1}

    v1, v2, vc = vec(pair1), vec(pair2), vec(cross)
    currencies = set(v1) | set(v2) | set(vc)
    for c1 in (1, -1):
        for c2 in (1, -1):
            if all(c1 * v1.get(ccy, 0) + c2 * v2.get(ccy, 0) == vc.get(ccy, 0) for ccy in currencies):
                return c1, c2
    raise ValueError(
        f"cannot triangulate {cross} from {pair1} and {pair2}: no +/-1 combination "
        "of the two legs cancels the common currency"
    )


def triangulate_prices(
    prices1: Sequence[float] | np.ndarray | pd.Series,
    pair1: str,
    prices2: Sequence[float] | np.ndarray | pd.Series,
    pair2: str,
    cross: str,
) -> np.ndarray:
    """Build a cross rate series ``S_cross = S_1^{c1} * S_2^{c2}``.

    E.g. EURJPY = EURUSD * USDJPY, or EURJPY = EURUSD / JPYUSD.
    """
    c1, c2 = cross_pair_signs(pair1, pair2, cross)
    p1 = _validate_prices(prices1, "prices1")
    p2 = _validate_prices(prices2, "prices2")
    if p1.shape != p2.shape:
        raise ValueError(f"leg price series must align, got lengths {p1.size} and {p2.size}")
    return p1 ** c1 * p2 ** c2


def triangulate_returns(
    returns1: Sequence[float] | np.ndarray | pd.Series,
    pair1: str,
    returns2: Sequence[float] | np.ndarray | pd.Series,
    pair2: str,
    cross: str,
) -> np.ndarray:
    """Cross log returns ``r_cross = c1 r_1 + c2 r_2`` (exact for log returns)."""
    c1, c2 = cross_pair_signs(pair1, pair2, cross)
    r1 = _as_1d_array(returns1, "returns1")
    r2 = _as_1d_array(returns2, "returns2")
    if r1.shape != r2.shape:
        raise ValueError(f"leg return series must align, got lengths {r1.size} and {r2.size}")
    return c1 * r1 + c2 * r2


def cross_volatility(
    vol1: float,
    vol2: float,
    corr: float,
    pair1: str | None = None,
    pair2: str | None = None,
    cross: str | None = None,
    sign_product: int | None = None,
) -> float:
    """Volatility of a triangulated cross from leg vols and their correlation.

    ``sigma_cross^2 = sigma_1^2 + sigma_2^2 + 2 * s * rho * sigma_1 * sigma_2``
    where ``s = c1 * c2`` is the product of the quote-direction signs.

    Parameters
    ----------
    vol1, vol2 : float
        Leg volatilities (any common unit: daily or annualized, decimal or
        percent -- the output inherits the unit).
    corr : float
        Correlation of the two legs' log returns *as quoted*, in [-1, 1].
    pair1, pair2, cross : str, optional
        If given, the sign product is derived from the pair strings via
        :func:`cross_pair_signs`.
    sign_product : int, optional
        Directly supply ``c1 * c2`` (+1 or -1) instead of pair strings.

    Returns
    -------
    float
        The cross volatility, same unit as the inputs.
    """
    for name, value in (("vol1", vol1), ("vol2", vol2), ("corr", corr)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if vol1 < 0 or vol2 < 0:
        raise ValueError("volatilities must be non-negative")
    if not -1.0 <= corr <= 1.0:
        raise ValueError(f"correlation must lie in [-1, 1], got {corr}")
    if sign_product is None:
        if pair1 is None or pair2 is None or cross is None:
            raise ValueError("supply either sign_product or all of pair1/pair2/cross")
        c1, c2 = cross_pair_signs(pair1, pair2, cross)
        sign_product = c1 * c2
    if sign_product not in (1, -1):
        raise ValueError(f"sign_product must be +1 or -1, got {sign_product}")
    var = vol1 ** 2 + vol2 ** 2 + 2.0 * sign_product * corr * vol1 * vol2
    # numerical guard: exact cancellation (rho = -1, equal vols) can dip below 0
    return float(np.sqrt(max(var, 0.0)))
