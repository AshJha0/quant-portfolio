"""NumPy array entry points for Black-Scholes-Merton prices and Greeks.

The scalar functions in :mod:`eq_options.black_scholes` and
:mod:`eq_options.greeks` take one option at a time and stay the reference.
These functions take arrays (or anything ``numpy.broadcast_arrays`` accepts),
validate every element, and price the whole book with NumPy ufuncs.

Everything after ``K`` is keyword-only. ``T``, ``r`` and ``sigma`` are all
plain floats, so a positional call that swaps two of them runs silently and
returns wrong numbers; naming them removes that failure mode.

Agreement with the scalar API is a tested contract, not bit identity:
``tests/test_arrays.py`` checks ``|array - scalar| <= 1e-12 * max(1, S, K)``
over a grid that includes every boundary branch (T, K, S or sigma zero).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qp_core.black_scholes import carry_greeks_array, carry_price_array

__all__ = ["bs_greeks_array", "bs_price_array"]

def _first_bad(mask: NDArray[np.bool_]) -> int:
    return int(np.flatnonzero(mask.ravel())[0])


_F64 = NDArray[np.float64]


def _prepare(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike,
    option_type: str | ArrayLike,
    *,
    strict: bool,
) -> tuple[_F64, _F64, _F64, _F64, _F64, _F64, NDArray[np.bool_]]:
    arrs = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, K, T, r, sigma, q)))
    S, K, T, r, sigma, q = (np.ascontiguousarray(a) for a in arrs)
    ot = np.asarray(option_type)
    if ot.dtype.kind not in "US":
        raise ValueError("option_type must be 'call', 'put' or an array of those strings")
    ot = np.char.lower(ot.astype(str))
    bad_ot = (ot != "call") & (ot != "put")
    if bad_ot.any():
        i = _first_bad(bad_ot)
        raise ValueError(f"option_type must be 'call' or 'put', got {ot.ravel()[i]!r} at index {i}")
    is_call = np.broadcast_to(ot == "call", S.shape)
    for name, a in (("S", S), ("K", K), ("T", T), ("sigma", sigma)):
        bad = ~np.isfinite(a) | (a <= 0.0 if strict else a < 0.0)
        if bad.any():
            i = _first_bad(bad)
            rule = "finite and > 0" if strict else "finite and >= 0"
            raise ValueError(f"{name} must be {rule}, got {a.ravel()[i]!r} at index {i}")
    for name, a in (("r", r), ("q", q)):
        bad = ~np.isfinite(a)
        if bad.any():
            i = _first_bad(bad)
            raise ValueError(f"{name} must be finite, got {a.ravel()[i]!r} at index {i}")
    return S, K, T, r, sigma, q, is_call


def _check_finite(name: str, a: NDArray[np.float64]) -> None:
    bad = ~np.isfinite(a)
    if bad.any():
        raise ValueError(f"{name} is not representable as a finite double at index {_first_bad(bad)}")


def bs_price_array(
    S: ArrayLike,
    K: ArrayLike,
    *,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str | ArrayLike = "call",
) -> NDArray[np.float64]:
    """Black-Scholes-Merton prices for broadcastable arrays of inputs.

    Same domain and the same boundary rules as :func:`bs_price` (``T == 0``
    gives intrinsic, ``K == 0`` a discounted forward for a call, ``S == 0``
    a discounted strike for a put, ``sigma == 0`` the discounted forward
    intrinsic). ``option_type`` is ``"call"``, ``"put"`` or an array of
    those strings.

    Raises
    ------
    ValueError
        Naming the field and the first flat index that fails validation, or
        the first index whose price overflows.
    """
    S, K, T, r, sigma, q, is_call = _prepare(S, K, T, r, sigma, q, option_type, strict=False)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        disc_s = S * np.exp(-q * T)
        disc_k = K * np.exp(-r * T)
        general = (T > 0.0) & (K > 0.0) & (S > 0.0) & (sigma > 0.0)
        # Harmless stand-ins where the general formula does not apply.
        Sg, Kg = np.where(general, S, 1.0), np.where(general, K, 1.0)
        Tg, vg = np.where(general, T, 1.0), np.where(general, sigma, 1.0)
        sign = np.where(is_call, 1.0, -1.0)
        # Shared carry-form kernel (qp_core.black_scholes); + 0.0 keeps the
        # +0.0 a put worth exactly zero has always returned.
        bs = carry_price_array(Sg, Kg, Tg, r, q, vg, sign, disc_s, disc_k) + 0.0
        price = np.select(
            [T == 0.0, K == 0.0, S == 0.0, sigma == 0.0],
            [
                np.maximum(sign * (S - K), 0.0),
                np.where(is_call, disc_s, 0.0),
                np.where(is_call, 0.0, disc_k),
                np.maximum(sign * (disc_s - disc_k), 0.0),
            ],
            default=bs,
        )
    _check_finite("bs_price", price)
    return price


def bs_greeks_array(
    S: ArrayLike,
    K: ArrayLike,
    *,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str | ArrayLike = "call",
) -> dict[str, NDArray[np.float64]]:
    """Price and analytic Greeks for arrays, as a dict of arrays.

    Keys and units are those of :class:`BSGreeks` (price, delta, gamma, vega,
    theta, rho, vanna, volga). Like :func:`bs_greeks`, ``S``, ``K``, ``T``
    and ``sigma`` must be strictly positive.
    """
    S, K, T, r, sigma, q, is_call = _prepare(S, K, T, r, sigma, q, option_type, strict=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        g = carry_greeks_array(S, K, T, r, q, sigma, np.where(is_call, 1.0, -1.0))
        keys = ("price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga")
        out = {k: g[k] for k in keys}
        # + 0.0: the put price this function used to evaluate (K e^{-rT}
        # N(-d2) - S e^{-qT} N(-d1)) never returned -0.0; the phi form can.
        out["price"] = out["price"] + 0.0
    for name, a in out.items():
        _check_finite(f"bs_greeks.{name}", a)
    return out
