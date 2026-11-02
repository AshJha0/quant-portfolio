"""NumPy array entry points for Garman-Kohlhagen prices and Greeks.

:func:`fx_options.garman_kohlhagen.gk_price` prices one option and stays the
reference. :func:`gk_price_array` takes broadcastable arrays, validates every
element and prices the book with NumPy ufuncs. Everything after ``K`` is
keyword-only, because ``T``, ``r_d``, ``r_f`` and ``sigma`` are all plain
floats and a swapped positional pair would run silently.

Agreement with the scalar API is tested, not bit identity:
``tests/test_arrays.py`` checks ``|array - scalar| <= 1e-12 * max(1, S, K)``,
including the ``T == 0`` and ``sigma sqrt(T) <= 1e-12`` branches.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erfc

from .garman_kohlhagen import _MIN_VOL

__all__ = ["gk_greeks_array", "gk_price_array"]

_RSQRT2 = 1.0 / np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _first_bad(mask: NDArray[np.bool_]) -> int:
    return int(np.flatnonzero(mask.ravel())[0])


def _prepare(S, K, T, r_d, r_f, sigma, option_type):
    arrs = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (S, K, T, r_d, r_f, sigma)))
    S, K, T, r_d, r_f, sigma = (np.ascontiguousarray(a) for a in arrs)
    ot = np.asarray(option_type)
    if ot.dtype.kind not in "US":
        raise ValueError("option_type must be 'call', 'put' or an array of those strings")
    ot = np.char.lower(ot.astype(str))
    bad_ot = (ot != "call") & (ot != "put")
    if bad_ot.any():
        i = _first_bad(bad_ot)
        raise ValueError(f"option_type must be 'call' or 'put', got {ot.ravel()[i]!r} at index {i}")
    phi = np.broadcast_to(np.where(ot == "call", 1.0, -1.0), S.shape)
    checks = (
        ("S", S, lambda a: a <= 0.0),
        ("K", K, lambda a: a <= 0.0),
        ("T", T, lambda a: a < 0.0),
        ("r_d", r_d, lambda a: np.zeros_like(a, bool)),
        ("r_f", r_f, lambda a: np.zeros_like(a, bool)),
        ("sigma", sigma, lambda a: a < 0.0),
    )
    for name, a, out_of_domain in checks:
        bad = ~np.isfinite(a) | out_of_domain(a)
        if bad.any():
            i = _first_bad(bad)
            raise ValueError(f"{name} is invalid ({a.ravel()[i]!r}) at index {i}")
    return S, K, T, r_d, r_f, sigma, phi


def gk_price_array(
    S: ArrayLike,
    K: ArrayLike,
    *,
    T: ArrayLike,
    r_d: ArrayLike,
    r_f: ArrayLike,
    sigma: ArrayLike,
    option_type: str | ArrayLike,
) -> NDArray[np.float64]:
    """Garman-Kohlhagen premiums (domestic per unit foreign) for arrays.

    Same domain and boundary rules as :func:`gk_price`: ``S, K > 0``,
    ``T, sigma >= 0``; ``T == 0`` gives intrinsic and ``sigma sqrt(T) <=
    1e-12`` the discounted forward intrinsic. ``option_type`` is ``"call"``,
    ``"put"`` or an array of those strings.

    Raises
    ------
    ValueError
        Naming the field and the first flat index that fails validation, or
        the first index whose premium overflows.
    """
    S, K, T, r_d, r_f, sigma, phi = _prepare(S, K, T, r_d, r_f, sigma, option_type)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        disc_s = S * np.exp(-r_f * T)
        disc_k = K * np.exp(-r_d * T)
        vst_raw = sigma * np.sqrt(T)
        general = vst_raw > _MIN_VOL
        Tg, vg = np.where(general, T, 1.0), np.where(general, sigma, 1.0)
        vst = vg * np.sqrt(Tg)
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * vg * vg) * Tg) / vst
        d2 = d1 - vst
        bs = phi * (disc_s * 0.5 * erfc(-phi * d1 * _RSQRT2) - disc_k * 0.5 * erfc(-phi * d2 * _RSQRT2))
        price = np.select(
            [T == 0.0, ~general],
            [np.maximum(phi * (S - K), 0.0), np.maximum(phi * (disc_s - disc_k), 0.0)],
            default=bs,
        )
    bad = ~np.isfinite(price)
    if bad.any():
        raise ValueError(f"gk_price is not representable as a finite double at index {_first_bad(bad)}")
    return price


def gk_greeks_array(
    S: ArrayLike,
    K: ArrayLike,
    *,
    T: ArrayLike,
    r_d: ArrayLike,
    r_f: ArrayLike,
    sigma: ArrayLike,
    option_type: str | ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Price and closed-form GK Greeks for arrays, as a dict of arrays.

    Keys and units are those of :class:`GreeksResult` (theta per year). Like
    :func:`analytic_greeks`, every element needs ``T > 0``, ``sigma > 0`` and
    ``sigma sqrt(T) > 1e-12``.
    """
    S, K, T, r_d, r_f, sigma, phi = _prepare(S, K, T, r_d, r_f, sigma, option_type)
    for name, a in (("T", T), ("sigma", sigma)):
        bad = a <= 0.0
        if bad.any():
            i = _first_bad(bad)
            raise ValueError(f"{name} must be > 0 for Greeks, got {a.ravel()[i]!r} at index {i}")
    bad = sigma * np.sqrt(T) <= _MIN_VOL
    if bad.any():
        raise ValueError(f"d1 undefined: sigma sqrt(T) <= {_MIN_VOL} at index {_first_bad(bad)}")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        sqrt_t = np.sqrt(T)
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        df_f = np.exp(-r_f * T)
        df_d = np.exp(-r_d * T)
        n_d1 = np.exp(-0.5 * d1 * d1) * _INV_SQRT_2PI
        n_pd1 = 0.5 * erfc(-phi * d1 * _RSQRT2)
        n_pd2 = 0.5 * erfc(-phi * d2 * _RSQRT2)
        vega = S * df_f * n_d1 * sqrt_t
        out = {
            "price": phi * (S * df_f * n_pd1 - K * df_d * n_pd2),
            "delta_spot": phi * df_f * n_pd1,
            "delta_forward": phi * n_pd1,
            "gamma": df_f * n_d1 / (S * sigma * sqrt_t),
            "vega": vega,
            "theta": -S * df_f * n_d1 * sigma / (2.0 * sqrt_t)
            + phi * (r_f * S * df_f * n_pd1 - r_d * K * df_d * n_pd2),
            "rho_domestic": phi * K * T * df_d * n_pd2,
            "rho_foreign": -phi * S * T * df_f * n_pd1,
            "vanna": -df_f * n_d1 * d2 / sigma,
            "volga": vega * d1 * d2 / sigma,
        }
    for name, a in out.items():
        bad = ~np.isfinite(a)
        if bad.any():
            raise ValueError(f"gk_greeks.{name} is not finite at index {_first_bad(bad)}")
    return out
