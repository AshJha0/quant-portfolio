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

Two-curve setup. The short rate drives the OIS discount curve ``D``. A
floating leg projected on a separate forward curve ``P_f`` uses the
deterministic multiplicative spread of Henrard (2010), "The irony in
derivatives discounting part II: the crisis", Wilmott Journal 2(6),
301-316, Section 3. For a float period ``[s_{j-1}, s_j]`` the ratio
``beta_j = (P_f(0,s_{j-1}) / P_f(0,s_j)) * (D(0,s_j) / D(0,s_{j-1}))`` is
held fixed, so at expiry ``T`` the float leg is worth
``sum_j (beta_j P(T,s_{j-1}) - P(T,s_j))``. The payer swaption is then a
put, struck at ``beta_1``, on a coupon bond with coefficient ``K tau`` on
fixed dates, ``1`` at the end and ``-(beta_{j+1} - 1)`` on float dates.
Jamshidian's split stays exact for the negative coefficients too: every
zero bond falls as ``r`` rises, so when the coupon bond crosses its strike
once (the root search checks this) all the pieces change sign together.
With ``forward_curve=None`` every ``beta_j = 1`` and the single-curve
formula comes back.

Calibration (``calibrate_hull_white``) fits ``(a, sigma)``, or ``sigma``
alone with ``a`` fixed, to a grid of European swaptions by least squares,
each swaption priced with the analytic Jamshidian formula (Brigo and
Mercurio 2006, Section 3.3.2). Market prices come from normal or Black
vols through the annuity formula in ``rate_options``. The tests check the
model against a Monte Carlo of the same dynamics and recover known
parameters from synthetic swaption grids.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
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

    def _swaption_bond(self, expiry: float, tenor: float, K: float, frequency: int,
                       forward_curve: DiscountCurve | None,
                       float_frequency: int | None) -> tuple[np.ndarray, np.ndarray, float]:
        """Coupon-bond times, coefficients and strike of the payer swaption."""
        n_fix = tenor * frequency
        if abs(n_fix - round(n_fix)) > 1e-9 or round(n_fix) < 1:
            raise ValueError("tenor * frequency must be a positive integer")
        ti = expiry + np.arange(1, int(round(n_fix)) + 1) / frequency
        coef: dict[float, float] = {}

        def add(t: float, c: float) -> None:
            key = round(float(t), 12)
            coef[key] = coef.get(key, 0.0) + c

        for t in ti:
            add(t, K / frequency)
        add(ti[-1], 1.0)
        strike = 1.0
        if forward_curve is not None:
            ff = frequency if float_frequency is None else float_frequency
            n_fl = tenor * ff
            if abs(n_fl - round(n_fl)) > 1e-9 or round(n_fl) < 1:
                raise ValueError("tenor * float_frequency must be a positive integer")
            s = expiry + np.arange(0, int(round(n_fl)) + 1) / ff
            pf = np.asarray(forward_curve.df(s), dtype=float)
            pd = np.asarray(self.curve.df(s), dtype=float)
            beta = (pf[:-1] / pf[1:]) * (pd[1:] / pd[:-1])
            strike = float(beta[0])
            for j in range(1, s.size - 1):
                add(s[j], -(beta[j] - 1.0))
        times = np.array(sorted(coef))
        return times, np.array([coef[t] for t in times]), strike

    def swaption(self, expiry: float, tenor: float, K: float, frequency: int = 1,
                 payer: bool = True, forward_curve: DiscountCurve | None = None,
                 float_frequency: int | None = None) -> float:
        """European swaption by Jamshidian decomposition (eq. 3.44-3.45).

        ``forward_curve`` (optional) projects the floating leg, paid
        ``float_frequency`` times a year (default ``frequency``), in the
        deterministic-spread two-curve setup of the module docstring.
        """
        ti, c, strike = self._swaption_bond(expiry, tenor, K, frequency,
                                            forward_curve, float_frequency)

        def g(r: float) -> float:
            return float(np.sum(c * self.zcb(expiry, ti, r))) - strike

        if not g(-1.0) > 0.0 > g(1.0):
            raise ValueError("coupon bond does not cross its strike on r in [-1, 1]")
        r_star = brentq(g, -1.0, 1.0, xtol=1e-14)
        Ki = self.zcb(expiry, ti, r_star)
        # payer swaption = put on the coupon bond struck at beta_1
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


# ------------------------------------------------------------ calibration
@dataclass(frozen=True)
class SwaptionQuote:
    """One market swaption. ``vol`` is normal (``model="normal"``) or Black.

    ``strike=None`` means at the money (the forward swap rate).
    """

    expiry: float
    tenor: float
    vol: float
    strike: float | None = None
    payer: bool = True
    model: str = "normal"
    fixed_frequency: int = 1
    float_frequency: int = 4


@dataclass(frozen=True)
class HullWhiteCalibration:
    model: HullWhite
    market_prices: np.ndarray
    model_prices: np.ndarray
    success: bool

    @property
    def max_abs_error(self) -> float:
        return float(np.max(np.abs(self.model_prices - self.market_prices)))


def calibrate_hull_white(
    curve: DiscountCurve,
    quotes: Sequence[SwaptionQuote],
    forward_curve: DiscountCurve | None = None,
    a: float | None = None,
    a0: float = 0.05,
    sigma0: float = 0.01,
) -> HullWhiteCalibration:
    """Fit Hull-White to a swaption grid by least squares on prices.

    ``curve`` is the OIS discount curve the model is fitted to;
    ``forward_curve`` (default: ``curve``) projects the floating legs. With
    ``a`` given only ``sigma`` is fitted, the usual desk choice because a
    single expiry column barely identifies ``a``. Residuals are price errors
    divided by the annuity (rate units), so long and short tenors weigh
    alike. Model prices use the analytic Jamshidian formula.
    """
    from .rate_options import forward_swap_rate, swap_annuity, swaption_price

    if len(quotes) == 0:
        raise ValueError("need at least one swaption quote")
    proj = curve if forward_curve is None else forward_curve
    strikes, mkt, ann = [], [], []
    for q in quotes:
        if q.model not in ("normal", "black"):
            raise ValueError(f"model must be 'normal' or 'black', got {q.model!r}")
        K = q.strike
        if K is None:
            K = forward_swap_rate(curve, proj, q.expiry, q.tenor,
                                  q.fixed_frequency, q.float_frequency)
        strikes.append(float(K))
        mkt.append(swaption_price(curve, proj, q.expiry, q.tenor, K, q.vol, q.payer,
                                  q.model, q.fixed_frequency, q.float_frequency))
        ann.append(swap_annuity(curve, q.expiry, q.tenor, q.fixed_frequency)[1])
    market = np.array(mkt)
    annuity = np.array(ann)

    def model_prices(av: float, sv: float) -> np.ndarray:
        hw = HullWhite(av, sv, curve)
        return np.array([hw.swaption(q.expiry, q.tenor, K, q.fixed_frequency, q.payer,
                                     forward_curve, q.float_frequency)
                         for q, K in zip(quotes, strikes, strict=True)])

    if a is None:
        def res(x: np.ndarray) -> np.ndarray:
            return (model_prices(float(np.exp(x[0])), float(np.exp(x[1]))) - market) / annuity
        sol = least_squares(res, np.log([a0, sigma0]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        av, sv = float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))
    else:
        def res1(x: np.ndarray) -> np.ndarray:
            return (model_prices(a, float(np.exp(x[0]))) - market) / annuity
        sol = least_squares(res1, np.log([sigma0]), xtol=1e-15, ftol=1e-15, gtol=1e-15)
        av, sv = a, float(np.exp(sol.x[0]))
    return HullWhiteCalibration(HullWhite(av, sv, curve), market, model_prices(av, sv),
                                bool(sol.success))
