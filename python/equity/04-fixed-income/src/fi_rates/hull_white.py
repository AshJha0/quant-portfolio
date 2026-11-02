"""One-factor Hull-White short-rate model, fitted to today's curve.

    dr = (theta(t) - a r) dt + sigma dW

All formulas from Brigo and Mercurio (2006), "Interest Rate Models - Theory
and Practice", 2nd ed., Section 3.3:

- ``r(t) = x(t) + alpha(t)``, ``dx = -a x dt + sigma dW``, ``x(0) = 0``,
  ``alpha(t) = f(0,t) + sigma^2/(2a^2) (1 - e^{-at})^2``      (eq. 3.36-3.37)
- ``P(t,T) = A(t,T) exp(-B(t,T) r(t))``, ``B = (1 - e^{-a(T-t)})/a``,
  ``A = P(0,T)/P(0,t) exp(B f(0,t) - sigma^2/(4a) (1 - e^{-2at}) B^2)``  (eq. 3.39)
- zero-bond option (eq. 3.40-3.41), with
  ``sigma_p = sigma B(T,S) sqrt((1 - e^{-2aT})/(2a))``
- caplet = ``(1 + K tau)`` puts on the zero bond struck at ``1/(1 + K tau)``
  (eq. 3.42)
- swaption by Jamshidian (1989) decomposition (eq. 3.44-3.45).

Single-curve: the model is fitted to one discount curve. The tests check
it against a Monte Carlo of the same dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .curve import DiscountCurve


@dataclass(frozen=True)
class HullWhite:
    a: float
    sigma: float
    curve: DiscountCurve

    def __post_init__(self) -> None:
        if self.a <= 0.0:
            raise ValueError(f"a must be > 0, got {self.a}")
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")

    # -------------------------------------------------------------- pieces
    def B(self, t: float, T: np.ndarray | float) -> np.ndarray | float:
        return (1.0 - np.exp(-self.a * (np.asarray(T) - t))) / self.a

    def inst_forward(self, t: float, h: float = 1e-5) -> float:
        """f(0,t) by central difference of ln P (one-sided at t ~ 0)."""
        lo = max(t - h, 0.0)
        return float(-(np.log(self.curve.df(t + h)) - np.log(self.curve.df(lo))) / (t + h - lo))

    def alpha(self, t: float) -> float:
        return self.inst_forward(t) + self.sigma**2 / (2 * self.a**2) * (1 - np.exp(-self.a * t)) ** 2

    def zcb(self, t: float, T: np.ndarray | float, r_t: np.ndarray | float) -> np.ndarray:
        """P(t,T) given the short rate r(t); broadcasts over ``T`` and ``r_t``."""
        T = np.asarray(T, dtype=float)
        b = self.B(t, T)
        p0T = np.asarray(self.curve.df(T))
        p0t = 1.0 if t <= 0 else float(self.curve.df(t))
        lnA = (np.log(p0T / p0t) + b * self.inst_forward(t)
               - self.sigma**2 / (4 * self.a) * (1 - np.exp(-2 * self.a * t)) * b * b)
        return np.exp(lnA - b * np.asarray(r_t, dtype=float))

    # ------------------------------------------------------------- options
    def bond_option(self, T: float, S: float, K: float, call: bool = True) -> float:
        """Option expiring at T on the zero bond maturing at S > T (eq. 3.40-3.41)."""
        if not 0.0 < T < S:
            raise ValueError("need 0 < T < S")
        sp = self.sigma * float(self.B(T, S)) * np.sqrt((1 - np.exp(-2 * self.a * T)) / (2 * self.a))
        pT, pS = float(self.curve.df(T)), float(self.curve.df(S))
        h = np.log(pS / (K * pT)) / sp + 0.5 * sp
        if call:
            return float(pS * norm.cdf(h) - K * pT * norm.cdf(h - sp))
        return float(K * pT * norm.cdf(-h + sp) - pS * norm.cdf(-h))

    def caplet(self, T: float, S: float, K: float, floorlet: bool = False) -> float:
        """Caplet on the simple rate over [T, S], unit notional, paid at S."""
        tau = S - T
        n = 1.0 + K * tau
        return n * self.bond_option(T, S, 1.0 / n, call=floorlet)

    def cap(self, maturity: float, K: float, frequency: int = 4, floor: bool = False) -> float:
        t = np.arange(0, int(round(maturity * frequency)) + 1) / frequency
        return float(sum(self.caplet(t0, t1, K, floor) for t0, t1 in zip(t[1:-1], t[2:], strict=False)))

    def swaption(self, expiry: float, tenor: float, K: float, frequency: int = 1,
                 payer: bool = True) -> float:
        """European swaption by Jamshidian decomposition (eq. 3.44-3.45)."""
        ti = expiry + np.arange(1, int(round(tenor * frequency)) + 1) / frequency
        c = np.full(ti.size, K / frequency)
        c[-1] += 1.0

        def g(r: float) -> float:
            return float(np.sum(c * self.zcb(expiry, ti, r))) - 1.0

        r_star = brentq(g, -1.0, 1.0, xtol=1e-14)
        Ki = self.zcb(expiry, ti, r_star)
        # payer swaption = put on the coupon bond struck at par
        return float(sum(ci * self.bond_option(expiry, s, k, call=not payer)
                         for ci, s, k in zip(c, ti, Ki, strict=False)))

    # ---------------------------------------------------------------- MC
    def simulate(self, T: float, n_steps: int, n_paths: int, seed: int = 0):
        """Exact OU steps for x; returns (r(T), integral of r over [0, T]) per path.

        The integral uses the trapezoid rule, so it carries an O(dt^2) bias.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        e = np.exp(-self.a * dt)
        sd = self.sigma * np.sqrt((1 - e * e) / (2 * self.a))
        half = n_paths // 2
        x = np.zeros(2 * half)
        integral = np.zeros(2 * half)
        r_prev = x + self.alpha(0.0)
        for i in range(1, n_steps + 1):
            z = rng.standard_normal(half)
            x = x * e + sd * np.concatenate([z, -z])
            r = x + self.alpha(i * dt)
            integral += 0.5 * (r_prev + r) * dt
            r_prev = r
        return r_prev, integral
