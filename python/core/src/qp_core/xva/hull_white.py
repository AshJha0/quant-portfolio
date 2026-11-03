"""One-factor Hull-White model: bond formulas, Jamshidian swaptions, exact MC.

Plain idea
----------
To price counterparty risk on a swap we need many possible futures of the
yield curve, not one. Hull-White is the simplest model that both reproduces
today's curve exactly and lets rates wander with a chosen volatility. Its
bond prices are closed form, so a European swaption (the option to enter
the swap) also has a near-closed form, and that gives an exact check on the
Monte Carlo exposure.

Formal statement
----------------
Short rate r(t) = x(t) + phi(t), dx = -a x dt + sigma dW under the bank
account measure, x(0) = 0, phi fitted to the initial discount curve P(0, .).
With B(t,T) = (1 - e^{-a(T-t)}) / a and

    V(t,T) = sigma^2 / a^2 [ (T-t) + 2 e^{-a(T-t)}/a - e^{-2a(T-t)}/(2a) - 3/(2a) ],

the bond price is P(t,T) = A(t,T) exp(-B(t,T) x(t)) with
A(t,T) = P(0,T)/P(0,t) exp((V(t,T) - V(0,T) + V(0,t)) / 2)
(Brigo and Mercurio 2006, eq. 3.39). Zero-coupon bond options follow
eq. 3.40, and a coupon-bond option (a swaption) splits into a sum of them by
Jamshidian (1989): solve sum c_i P(T0, T_i; x*) = 1 for x*, set strikes
X_i = P(T0, T_i; x*).

Monte Carlo samples x(t) and I(t) = int_0^t x ds jointly and exactly (they
are Gaussian) on a time grid; the discount factor is
D(0,t) = P(0,t) exp(-I(t) - V(0,t)/2), so E[D(0,t)] = P(0,t) to sampling
error with no time-step bias.

Worked example
--------------
>>> from qp_core.market import DiscountCurve
>>> hw = HullWhite(DiscountCurve((1.0, 30.0), (math.exp(-0.03), math.exp(-0.9))), a=0.05, sigma=0.01)
>>> round(hw.zcb(0.0, 5.0, 0.0), 10) == round(math.exp(-0.15), 10)
True
>>> round(hw.swaption(1.0, [2.0, 3.0, 4.0, 5.0], [1.0] * 4, 0.03, payer=True) * 1e4, 2)
139.53

Sources: Brigo and Mercurio, *Interest Rate Models*, 2nd ed. (2006),
ch. 3.3; Jamshidian, *Journal of Finance* 44 (1989); Glasserman,
*Monte Carlo Methods in Financial Engineering* (2003), sec. 3.3.2 for the
exact joint (x, I) step. Evidence class: textbook closed forms, checked
against the Monte Carlo in the tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..errors import ConvergenceError, InputError
from ..market import DiscountCurve
from ..normal import norm_cdf

__all__ = ["HullWhite", "HullWhitePaths"]


@dataclass(frozen=True)
class HullWhitePaths:
    """Simulated state on ``times`` (``times[0] == 0``): ``x[p, k]`` and the
    pathwise discount factor ``disc[p, k] = D(0, times[k])``."""

    times: np.ndarray
    x: np.ndarray
    disc: np.ndarray


class HullWhite:
    """Hull-White one-factor model fitted to ``curve`` with mean reversion
    ``a`` > 0 and volatility ``sigma`` > 0."""

    def __init__(self, curve: DiscountCurve, a: float, sigma: float) -> None:
        if not (a > 0.0 and math.isfinite(a)) or not (sigma > 0.0 and math.isfinite(sigma)):
            raise InputError("a and sigma must be positive and finite")
        self.curve, self.a, self.sigma = curve, float(a), float(sigma)

    def p0(self, t: float) -> float:
        return 1.0 if t <= 0.0 else self.curve.df(t)

    def b(self, t: float, T: float) -> float:
        return (1.0 - math.exp(-self.a * (T - t))) / self.a

    def v(self, t: float, T: float) -> float:
        a, s, tau = self.a, self.sigma, T - t
        return (s * s / (a * a)) * (
            tau + 2.0 * math.exp(-a * tau) / a - math.exp(-2.0 * a * tau) / (2.0 * a) - 1.5 / a
        )

    def log_a(self, t: float, T: float) -> float:
        return math.log(self.p0(T) / self.p0(t)) + 0.5 * (self.v(t, T) - self.v(0.0, T) + self.v(0.0, t))

    def zcb(self, t: float, T: float, x: float) -> float:
        """P(t, T) given x(t) = x."""
        return math.exp(self.log_a(t, T) - self.b(t, T) * x)

    def _sigma_p(self, T: float, S: float) -> float:
        a = self.a
        return self.sigma * math.sqrt((1.0 - math.exp(-2.0 * a * T)) / (2.0 * a)) * self.b(T, S)

    def zcb_option(self, T: float, S: float, strike: float, *, call: bool) -> float:
        """Time-0 price of an option expiring T on the bond maturing S > T (BM eq. 3.40)."""
        if not (0.0 < T < S) or not strike > 0.0:
            raise InputError("need 0 < T < S and strike > 0")
        sp = self._sigma_p(T, S)
        pt, ps = self.p0(T), self.p0(S)
        h = math.log(ps / (pt * strike)) / sp + 0.5 * sp
        if call:
            return ps * norm_cdf(h) - strike * pt * norm_cdf(h - sp)
        return strike * pt * norm_cdf(-h + sp) - ps * norm_cdf(-h)

    def swaption(
        self,
        expiry: float,
        pay_times: Sequence[float],
        accruals: Sequence[float],
        strike: float,
        *,
        payer: bool,
    ) -> float:
        """Time-0 price, per unit notional, of a European swaption on the swap
        starting at ``expiry`` with fixed payments ``strike * accruals[i]`` at
        ``pay_times[i]`` (single-curve), by Jamshidian's decomposition.

        A payer swaption is a put on the coupon bond, a receiver a call.
        """
        if len(pay_times) != len(accruals) or not pay_times:
            raise InputError("pay_times and accruals must be non-empty and equal length")
        if not all(t > expiry for t in pay_times) or expiry <= 0.0:
            raise InputError("pay times must follow expiry > 0")
        c = [strike * tau for tau in accruals]
        c[-1] += 1.0
        xstar = self._jamshidian_root(expiry, pay_times, c)
        total = 0.0
        for ci, ti in zip(c, pay_times, strict=True):
            xi = self.zcb(expiry, ti, xstar)
            total += ci * self.zcb_option(expiry, ti, xi, call=not payer)
        return total

    def _jamshidian_root(self, T0: float, times: Sequence[float], c: Sequence[float]) -> float:
        """Solve sum c_i P(T0, T_i; x) = 1 for x by bisection (the sum is
        strictly decreasing in x when every c_i > 0). Fixed 200 halvings so
        the Rust and C++ ports reproduce the same root."""
        if any(ci <= 0.0 for ci in c):
            raise InputError("Jamshidian needs positive coupons")

        def f(x: float) -> float:
            return sum(ci * self.zcb(T0, ti, x) for ci, ti in zip(c, times, strict=True)) - 1.0

        lo, hi = -1.0, 1.0
        k = 0
        while f(lo) < 0.0:
            lo *= 2.0
            k += 1
            if k > 60:
                raise ConvergenceError("Jamshidian bracket failed")
        while f(hi) > 0.0:
            hi *= 2.0
            k += 1
            if k > 60:
                raise ConvergenceError("Jamshidian bracket failed")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid in (lo, hi):
                break
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def simulate(
        self, times: Sequence[float], n_paths: int, seed: int, *, antithetic: bool = True
    ) -> HullWhitePaths:
        """Exact joint simulation of (x, I) on ``times`` (strictly increasing,
        starting at 0). Seeded numpy PCG64; antithetic pairs halve variance on
        the linear part of the payoff."""
        t = np.asarray(times, dtype=float)
        if t.ndim != 1 or t.size < 2 or t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
            raise InputError("times must start at 0 and increase strictly")
        if n_paths < 2 or (antithetic and n_paths % 2):
            raise InputError("n_paths must be >= 2 (and even with antithetic)")
        rng = np.random.default_rng(seed)
        half = n_paths // 2 if antithetic else n_paths
        a, s = self.a, self.sigma
        x = np.zeros((n_paths, t.size))
        integ = np.zeros(n_paths)
        disc = np.ones((n_paths, t.size))
        for k in range(1, t.size):
            dt = t[k] - t[k - 1]
            e1, e2 = math.exp(-a * dt), math.exp(-2.0 * a * dt)
            var_x = s * s * (1.0 - e2) / (2.0 * a)
            var_i = (s * s / (a * a)) * (dt - 2.0 * (1.0 - e1) / a + (1.0 - e2) / (2.0 * a))
            cov = (s * s / (2.0 * a * a)) * (1.0 - e1) ** 2
            l11 = math.sqrt(var_x)
            l21 = cov / l11
            l22 = math.sqrt(max(var_i - l21 * l21, 0.0))
            z = rng.standard_normal((half, 2))
            if antithetic:
                z = np.concatenate([z, -z])
            xp = x[:, k - 1]
            integ = integ + xp * (1.0 - e1) / a + l21 * z[:, 0] + l22 * z[:, 1]
            x[:, k] = xp * e1 + l11 * z[:, 0]
            disc[:, k] = self.p0(float(t[k])) * np.exp(-integ - 0.5 * self.v(0.0, float(t[k])))
        return HullWhitePaths(t, x, disc)
