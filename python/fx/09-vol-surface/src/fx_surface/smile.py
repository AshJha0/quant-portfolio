"""Smile models: SVI in log-moneyness and the market-standard vanna-volga.

Two complementary fits through the five pillar vols:

* :class:`SVISmile` - Gatheral's raw SVI parameterisation of *total
  variance* in log-moneyness ``k = ln(K/F)``.  Smooth, globally
  well-behaved wings (linear total variance), and the Durrleman
  butterfly-arbitrage condition can be checked with analytic
  derivatives.
* :class:`VannaVolgaSmile` - the FX market's workhorse: the price of any
  strike is the flat-ATM Black-Scholes price plus the cost of the
  three-instrument (25P / ATM / 25C) hedge portfolio that matches the
  target option's vega, vanna and volga at the reference (ATM) vol.
  This is the *exact replication-weight* construction (a 3x3 linear
  solve per strike), not the first-order approximation.

Vanna-volga is exact at the three pillars by construction, cheap and
desk-standard for barriers/touches, but it is an interpolation device,
not a model: the wings beyond 10-delta follow the extrapolated quadratic
vol-of-vol cost and can violate no-arbitrage far out.  SVI has
disciplined wings but no pricing dynamics.  Both are fitted and compared
in the pipeline; Heston sits on top as the dynamics model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .garman_kohlhagen import (
    gk_forward,
    gk_price,
    gk_vanna,
    gk_vega,
    gk_volga,
    implied_vol,
)

__all__ = [
    "SVIParams",
    "SVISmile",
    "VannaVolgaSmile",
    "durrleman_g",
    "smile_digital",
]


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI: ``w(k) = a + b [rho (k - m) + sqrt((k - m)^2 + s^2)]``."""

    a: float
    b: float
    rho: float
    m: float
    s: float

    def __post_init__(self) -> None:
        if self.b < 0.0:
            raise ValueError(f"SVI b must be >= 0, got {self.b}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"SVI rho must be in (-1, 1), got {self.rho}")
        if self.s <= 0.0:
            raise ValueError(f"SVI s must be > 0, got {self.s}")
        w_min = self.a + self.b * self.s * math.sqrt(max(1.0 - self.rho**2, 0.0))
        if w_min <= 0.0:
            raise ValueError(
                f"SVI minimum total variance {w_min} <= 0 (a={self.a}, b={self.b})"
            )


class SVISmile:
    """One-expiry SVI smile in log-moneyness ``k = ln(K/F)``.

    Parameters
    ----------
    params : SVIParams
    F : float
        Outright forward for the expiry.
    T : float
        Year fraction to expiry.
    """

    def __init__(self, params: SVIParams, F: float, T: float) -> None:
        if F <= 0.0 or T <= 0.0:
            raise ValueError("F and T must be positive")
        self.params = params
        self.F = F
        self.T = T

    # -- total variance and analytic derivatives ------------------------
    def total_variance(self, k: np.ndarray | float) -> np.ndarray | float:
        p = self.params
        km = np.asarray(k, dtype=float) - p.m
        return p.a + p.b * (p.rho * km + np.sqrt(km * km + p.s * p.s))

    def w_prime(self, k: np.ndarray | float) -> np.ndarray | float:
        p = self.params
        km = np.asarray(k, dtype=float) - p.m
        return p.b * (p.rho + km / np.sqrt(km * km + p.s * p.s))

    def w_second(self, k: np.ndarray | float) -> np.ndarray | float:
        p = self.params
        km = np.asarray(k, dtype=float) - p.m
        return p.b * p.s * p.s / np.power(km * km + p.s * p.s, 1.5)

    # -- vol queries ----------------------------------------------------
    def vol_logm(self, k: np.ndarray | float) -> np.ndarray | float:
        """Implied vol at log-moneyness ``k = ln(K/F)``."""
        return np.sqrt(self.total_variance(k) / self.T)

    def vol(self, K: np.ndarray | float) -> np.ndarray | float:
        """Implied vol at strike ``K``."""
        return self.vol_logm(np.log(np.asarray(K, dtype=float) / self.F))

    # -- arbitrage ------------------------------------------------------
    def durrleman_g(self, k: np.ndarray | float) -> np.ndarray | float:
        """Durrleman's density factor; ``g >= 0`` iff no butterfly arb."""
        w = self.total_variance(k)
        wp = self.w_prime(k)
        wpp = self.w_second(k)
        k = np.asarray(k, dtype=float)
        return (1.0 - k * wp / (2.0 * w)) ** 2 - 0.25 * wp * wp * (1.0 / w + 0.25) + 0.5 * wpp

    def is_butterfly_arbitrage_free(
        self, k_lo: float = -1.5, k_hi: float = 1.5, n: int = 601
    ) -> tuple[bool, float]:
        """Grid check of Durrleman's condition.

        Returns ``(ok, min_g)``; ``ok`` is True when ``min_g >= -1e-10``.
        """
        g = self.durrleman_g(np.linspace(k_lo, k_hi, n))
        g_min = float(np.min(g))
        return g_min >= -1e-10, g_min

    # -- fitting --------------------------------------------------------
    @classmethod
    def fit(
        cls, strikes: np.ndarray, vols: np.ndarray, F: float, T: float
    ) -> "SVISmile":
        """Least-squares fit of raw SVI through (strikes, vols).

        Five points / five parameters: for clean data this is (near-)
        exact interpolation.  Multi-start trust-region-reflective with
        bounds; a flat input smile (max-min vol < 1e-8) short-circuits
        to the degenerate flat SVI (``b = 0``), which is otherwise an
        unidentifiable ridge in (rho, m, s).
        """
        strikes = np.asarray(strikes, dtype=float)
        vols = np.asarray(vols, dtype=float)
        if strikes.shape != vols.shape or strikes.ndim != 1 or len(strikes) < 3:
            raise ValueError("need matching 1-D strikes/vols with >= 3 points")
        if np.any(vols <= 0.0):
            raise ValueError("vols must be positive")
        k = np.log(strikes / F)
        w_target = vols * vols * T

        if float(np.max(vols) - np.min(vols)) < 1e-8:  # flat smile: degenerate SVI
            params = SVIParams(a=float(np.mean(w_target)), b=0.0, rho=0.0, m=0.0, s=0.1)
            return cls(params, F, T)

        w_min, w_max = float(np.min(w_target)), float(np.max(w_target))
        k_span = float(np.max(k) - np.min(k))

        def residuals(x: np.ndarray) -> np.ndarray:
            a, b, rho, m, s = x
            km = k - m
            w = a + b * (rho * km + np.sqrt(km * km + s * s))
            return w - w_target

        lb = [1e-12, 1e-12, -0.9999, -2.0, 1e-4]
        ub = [2.0 * w_max, 10.0 * w_max / max(k_span, 1e-2), 0.9999, 2.0, 5.0]
        skew_sign = math.copysign(1.0, w_target[-1] - w_target[0])
        starts = [
            [0.8 * w_min, (w_max - w_min) / max(k_span, 1e-2), 0.4 * skew_sign, 0.0, 0.2],
            [0.5 * w_min, 2.0 * (w_max - w_min) / max(k_span, 1e-2), -0.5, 0.05, 0.1],
            [0.9 * w_min, 0.5 * (w_max - w_min) / max(k_span, 1e-2), 0.5, -0.05, 0.4],
        ]
        best = None
        for x0 in starts:
            x0 = np.clip(x0, lb, ub)
            try:
                res = least_squares(
                    residuals, x0, bounds=(lb, ub), xtol=1e-15, ftol=1e-15, gtol=1e-15
                )
            except ValueError:
                continue
            if best is None or res.cost < best.cost:
                best = res
            if best.cost < 1e-20:
                break
        if best is None:
            raise RuntimeError("SVI fit failed from all starts")
        a, b, rho, m, s = best.x
        # Keep parameters strictly inside the valid region.
        b = max(b, 0.0)
        params = SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m), s=float(s))
        return cls(params, F, T)


class VannaVolgaSmile:
    """Exact-weight vanna-volga smile from the 25P / ATM / 25C pillars.

    For a target strike K the hedge weights ``x = (x_1, x_2, x_3)``
    solve the 3x3 replication system (all Greeks at the reference ATM
    vol ``sigma_2``):

        sum_i x_i vega(K_i)  = vega(K)
        sum_i x_i vanna(K_i) = vanna(K)
        sum_i x_i volga(K_i) = volga(K)

    and the VV price is ``BS(K, sigma_2) + sum_i x_i [BS(K_i, sigma_i) -
    BS(K_i, sigma_2)]`` - the Black-Scholes price plus the smile cost of
    the replicating pillar portfolio.  At ``K = K_i`` the system returns
    the unit vector, so pillar prices (and vols) are reproduced exactly.

    Parameters
    ----------
    S, T, r_d, r_f : floats
        Market inputs for the expiry.
    pillar_strikes, pillar_vols : arrays of length 3
        (K_25P, K_ATM, K_25C) and their vols; strikes strictly increasing.
    """

    def __init__(
        self,
        S: float,
        T: float,
        r_d: float,
        r_f: float,
        pillar_strikes: np.ndarray,
        pillar_vols: np.ndarray,
    ) -> None:
        pillar_strikes = np.asarray(pillar_strikes, dtype=float)
        pillar_vols = np.asarray(pillar_vols, dtype=float)
        if pillar_strikes.shape != (3,) or pillar_vols.shape != (3,):
            raise ValueError("vanna-volga needs exactly 3 pillars (25P, ATM, 25C)")
        if not (pillar_strikes[0] < pillar_strikes[1] < pillar_strikes[2]):
            raise ValueError(f"pillar strikes must be increasing: {pillar_strikes}")
        if np.any(pillar_vols <= 0.0):
            raise ValueError("pillar vols must be positive")
        self.S, self.T, self.r_d, self.r_f = S, T, r_d, r_f
        self.F = gk_forward(S, T, r_d, r_f)
        self.Ks = pillar_strikes
        self.sigmas = pillar_vols
        self.sigma_ref = float(pillar_vols[1])  # ATM as the reference vol
        # Pillar Greek matrix at the reference vol (columns = pillars).
        self._A = np.array(
            [
                [gk_vega(S, K, T, r_d, r_f, self.sigma_ref) for K in self.Ks],
                [gk_vanna(S, K, T, r_d, r_f, self.sigma_ref) for K in self.Ks],
                [gk_volga(S, K, T, r_d, r_f, self.sigma_ref) for K in self.Ks],
            ]
        )
        if abs(np.linalg.det(self._A)) < 1e-16:
            raise ValueError("degenerate pillar configuration: singular VV system")
        # Smile cost of each pillar: market price minus flat-ATM BS price.
        self._pillar_cost = np.array(
            [
                gk_price(S, K, T, r_d, r_f, sig, +1)
                - gk_price(S, K, T, r_d, r_f, self.sigma_ref, +1)
                for K, sig in zip(self.Ks, self.sigmas)
            ]
        )

    def weights(self, K: float) -> np.ndarray:
        """Replication weights x(K) solving the 3x3 vega/vanna/volga system."""
        b = np.array(
            [
                gk_vega(self.S, K, self.T, self.r_d, self.r_f, self.sigma_ref),
                gk_vanna(self.S, K, self.T, self.r_d, self.r_f, self.sigma_ref),
                gk_volga(self.S, K, self.T, self.r_d, self.r_f, self.sigma_ref),
            ]
        )
        return np.linalg.solve(self._A, b)

    def price(self, K: float, cp: int = 1) -> float:
        """Vanna-volga price.  Put prices follow from the call via parity
        (the VV adjustment is strike-level, so parity is preserved)."""
        x = self.weights(K)
        call = gk_price(self.S, K, self.T, self.r_d, self.r_f, self.sigma_ref, +1)
        call += float(x @ self._pillar_cost)
        if cp == 1:
            return call
        return call - math.exp(-self.r_d * self.T) * (self.F - K)

    def vol(self, K: np.ndarray | float) -> np.ndarray | float:
        """VV implied vol at strike(s): invert the VV call price."""
        if np.ndim(K) > 0:
            return np.array([self.vol(float(x)) for x in np.asarray(K, dtype=float)])
        price = self.price(float(K), +1)
        return implied_vol(
            price, self.S, float(K), self.T, self.r_d, self.r_f, +1, on_fail="nan"
        )


def durrleman_g(
    vol_fn, F: float, T: float, k: np.ndarray, h: float = 1e-4
) -> np.ndarray:
    """Durrleman density factor for a generic smile ``sigma(K)`` via FD.

    ``vol_fn`` maps strike -> vol.  Total variance derivatives are
    central finite differences in ``k`` with step ``h``.  ``g >= 0``
    everywhere is equivalent to a non-negative implied density
    (no butterfly arbitrage).
    """
    k = np.asarray(k, dtype=float)

    def w(kk: np.ndarray) -> np.ndarray:
        sig = np.array([vol_fn(F * math.exp(x)) for x in np.atleast_1d(kk)])
        return sig * sig * T

    w0, wp_ = w(k), (w(k + h) - w(k - h)) / (2.0 * h)
    wpp = (w(k + h) - 2.0 * w0 + w(k - h)) / (h * h)
    return (1.0 - k * wp_ / (2.0 * w0)) ** 2 - 0.25 * wp_ * wp_ * (1.0 / w0 + 0.25) + 0.5 * wpp


def smile_digital(
    smile, K: float, S: float, T: float, r_d: float, r_f: float, cp: int = 1,
    h_rel: float = 1e-4,
) -> float:
    """Smile-consistent domestic cash digital via a tight call spread.

    ``digital_call = -dC/dK`` evaluated on the smile (so it includes the
    skew correction ``-vega_digital * dsigma/dK`` that the flat GK
    digital misses).  ``smile`` is any object with a ``vol(K)`` method.
    """
    h = h_rel * K
    c_lo = gk_price(S, K - h, T, r_d, r_f, float(smile.vol(K - h)), +1)
    c_hi = gk_price(S, K + h, T, r_d, r_f, float(smile.vol(K + h)), +1)
    dig_call = (c_lo - c_hi) / (2.0 * h)
    if cp == 1:
        return dig_call
    return math.exp(-r_d * T) - dig_call
