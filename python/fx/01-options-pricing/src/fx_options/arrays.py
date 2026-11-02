"""NumPy array entry point for Garman-Kohlhagen prices.

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

__all__ = ["gk_price_array"]

_RSQRT2 = 1.0 / np.sqrt(2.0)


def _first_bad(mask: NDArray[np.bool_]) -> int:
    return int(np.flatnonzero(mask.ravel())[0])


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
