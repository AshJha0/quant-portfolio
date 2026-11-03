"""SABR swaption cube: expiry x tenor x strike.

Plain idea. Swaption vols depend on three things: when the option expires,
how long the underlying swap runs, and the strike. Desks hold an ATM
matrix (expiry x tenor) and, at each node, a smile. Fitting SABR to the
smile at each node turns the cube into four small matrices of parameters
(``alpha``, ``rho``, ``nu``, with ``beta`` fixed), which interpolate far
more smoothly than raw vols and cannot produce a wild smile between nodes.

Formal statement.

* At node ``(T_e, T_n)`` the shifted-Black vol at strike ``K`` is
  ``sigma_SABR(F, K; T_e, alpha, beta, rho, nu, shift)`` (Hagan et al.
  2002, eq. 2.17a, with the Obloj 2008 leading term; :mod:`qp_core.vol.sabr`),
  where ``F`` is the forward swap rate. Calibration fits each node
  separately (:func:`qp_core.vol.sabr.sabr_calibrate`).
* Between nodes ``alpha``, ``rho`` and ``nu`` are interpolated bilinearly
  in ``(T_e, T_n)`` and held flat outside the grid. Interpolating
  parameters rather than vols is the design of QuantLib's SABR cube
  (``SabrSwaptionVolatilityCube``); it keeps the smile shape continuous. It does not guarantee absence of calendar arbitrage between
  nodes; that is stated, not checked.
* Price: ``A * Black(F + s, K + s, sigma, T_e)`` with annuity ``A`` from
  the discount curve (Brigo and Mercurio 2006, section 6.7.2;
  :func:`fi_rates.rate_options.swap_annuity`).

Data. No free swaption quotes exist (plan W15 risk). The sample cube in
``tests/test_swaptions.py`` is an illustrative surface: we chose its
SABR parameters (beta 0.5, rho from -0.45 to -0.15, nu from 0.25 to
0.55) by hand to look like a rates smile, and make no claim that they are
market levels; it is labelled
"illustrative, not market data" wherever it is used.

Worked example: a node fitted from its own SABR smile reprices every
strike to 1e-10 in vol, and a payer minus a receiver equals ``A (F - K)``
to rounding (put-call parity, model free).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from qp_core.vol.sabr import sabr_calibrate, sabr_vol

from .curve import DiscountCurve
from .rate_options import black76, forward_swap_rate, swap_annuity

__all__ = ["SabrSwaptionCube", "SmileQuote", "cube_swaption_price"]


@dataclass(frozen=True)
class SmileQuote:
    """One cube node: forward swap rate, strikes and shifted-Black vols."""

    forward: float
    strikes: tuple[float, ...]
    vols: tuple[float, ...]


def _interp2(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, x: float, y: float) -> float:
    x = min(max(x, xs[0]), xs[-1])
    y = min(max(y, ys[0]), ys[-1])
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, max(xs.size - 2, 0)))
    j = int(np.clip(np.searchsorted(ys, y) - 1, 0, max(ys.size - 2, 0)))
    if xs.size == 1:
        wx, i1 = 0.0, i
    else:
        wx, i1 = (x - xs[i]) / (xs[i + 1] - xs[i]), i + 1
    if ys.size == 1:
        wy, j1 = 0.0, j
    else:
        wy, j1 = (y - ys[j]) / (ys[j + 1] - ys[j]), j + 1
    return float((1 - wx) * (1 - wy) * z[i, j] + wx * (1 - wy) * z[i1, j]
                 + (1 - wx) * wy * z[i, j1] + wx * wy * z[i1, j1])


@dataclass(frozen=True)
class SabrSwaptionCube:
    """SABR parameters on an expiry x tenor grid (years); ``beta`` and ``shift`` global."""

    expiries: tuple[float, ...]
    tenors: tuple[float, ...]
    alpha: np.ndarray
    rho: np.ndarray
    nu: np.ndarray
    beta: float
    shift: float = 0.0
    rmse: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = (len(self.expiries), len(self.tenors))
        for name in ("alpha", "rho", "nu"):
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for axis in (self.expiries, self.tenors):
            if any(b <= a for a, b in zip(axis, axis[1:], strict=False)):
                raise ValueError("expiries and tenors must be strictly increasing")

    @classmethod
    def calibrate(cls, quotes: Mapping[tuple[float, float], SmileQuote], beta: float,
                  shift: float = 0.0) -> SabrSwaptionCube:
        """Fit SABR at every node of a full expiry x tenor grid of smiles."""
        ex = tuple(sorted({e for e, _ in quotes}))
        te = tuple(sorted({t for _, t in quotes}))
        if len(quotes) != len(ex) * len(te):
            raise ValueError("quotes must fill the expiry x tenor grid")
        a, r, n, rm = (np.empty((len(ex), len(te))) for _ in range(4))
        for i, e in enumerate(ex):
            for j, t in enumerate(te):
                q = quotes[(e, t)]
                fit = sabr_calibrate(q.forward, e, q.strikes, q.vols, beta, shift)
                a[i, j], r[i, j], n[i, j], rm[i, j] = fit.alpha, fit.rho, fit.nu, fit.rmse
        return cls(ex, te, a, r, n, beta, shift, rm)

    def params(self, expiry: float, tenor: float) -> tuple[float, float, float]:
        xs, ys = np.asarray(self.expiries), np.asarray(self.tenors)
        return (_interp2(xs, ys, np.asarray(self.alpha), expiry, tenor),
                _interp2(xs, ys, np.asarray(self.rho), expiry, tenor),
                _interp2(xs, ys, np.asarray(self.nu), expiry, tenor))

    def vol(self, expiry: float, tenor: float, strike: float, forward: float) -> float:
        """Shifted-Black vol at ``strike`` for the swap with this forward rate."""
        a, r, n = self.params(expiry, tenor)
        return sabr_vol(forward, strike, expiry, a, self.beta, r, n, self.shift)


def cube_swaption_price(cube: SabrSwaptionCube, discount: DiscountCurve,
                        projection: DiscountCurve, expiry: float, tenor: float,
                        strike: float, payer: bool = True, fixed_frequency: int = 1,
                        float_frequency: int = 4, notional: float = 1.0) -> float:
    """European swaption priced off the cube (shifted Black on the forward swap rate)."""
    _, A = swap_annuity(discount, expiry, tenor, fixed_frequency)
    F = forward_swap_rate(discount, projection, expiry, tenor, fixed_frequency, float_frequency)
    v = cube.vol(expiry, tenor, strike, F)
    return notional * A * black76(F + cube.shift, strike + cube.shift, expiry, v, call=payer)

