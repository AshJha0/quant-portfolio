"""Optimal market-making quotes: Avellaneda-Stoikov and Gueant-Lehalle-Fernandez-Tapia.

Plain idea
----------
A market maker posts a bid and an ask around the mid. Each fill earns half a
spread but moves inventory, and inventory loses money when the mid moves. A
risk-averse maker therefore (a) shades both quotes away from the side that
would grow inventory and (b) widens when there is more time left for the mid
to move against the position.

Formal statement
----------------
Both models share the same set-up (Avellaneda and Stoikov 2008, section 2;
Gueant, Lehalle and Fernandez-Tapia 2013, section 2):

* mid ``dS = sigma dW``;
* a quote at distance ``delta`` from the mid is hit with intensity
  ``A exp(-k delta)``;
* the maker maximises ``E[-exp(-gamma (X_T + q_T S_T))]`` (CARA, risk
  aversion ``gamma``).

**Avellaneda-Stoikov** (their equations 29-30, obtained from a quadratic
expansion of the value function in ``q``)::

    reservation price  r = s - q gamma sigma^2 (T - t)
    total spread       delta_a + delta_b = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k)

and the quotes are ``r -+ spread/2``.

**Gueant-Lehalle-Fernandez-Tapia** solve the same problem exactly when the
inventory is bounded, ``q in {-Q, ..., Q}``. With

    alpha = k gamma sigma^2 / 2,    eta = A (1 + gamma/k)^-(1 + k/gamma)

and ``M`` the ``(2Q+1)`` square tridiagonal matrix with ``alpha q^2`` on the
diagonal and ``-eta`` off it, ``v(t) = exp(-M (T-t)) 1`` and (their Theorem 1)::

    delta_b(t, q) = (1/k) ln(v_q / v_{q+1}) + (1/gamma) ln(1 + gamma/k)   for q < Q
    delta_a(t, q) = (1/k) ln(v_q / v_{q-1}) + (1/gamma) ln(1 + gamma/k)   for q > -Q

No bid is quoted at ``q = Q`` and no ask at ``q = -Q``: that is the inventory
bound. Far from the horizon ``v`` lines up with the ground eigenvector of
``M``, which gives stationary quotes. Their asymptotic approximation (section
4, used widely on desks because it needs no matrix) is::

    delta_b ~ (1/gamma) ln(1 + gamma/k) + (2q + 1)/2 * c
    delta_a ~ (1/gamma) ln(1 + gamma/k) - (2q - 1)/2 * c
    c = sqrt( sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma) )

Worked example
--------------
Avellaneda and Stoikov's own parameters (section 4): ``s = 100``,
``gamma = 0.1``, ``sigma = 2``, ``k = 1.5``, ``T - t = 1``. Then the spread is
``0.1 * 4 * 1 + 20 ln(1 + 0.1/1.5) = 0.4 + 1.2908 = 1.6908``; with ``q = 2``
the reservation price is ``100 - 2 * 0.1 * 4 = 99.2``, so the bid is
``98.3546`` and the ask ``100.0454``:

>>> p = ASParams(gamma=0.1, sigma=2.0, k=1.5)
>>> q = avellaneda_stoikov(100.0, 2, 1.0, p)
>>> round(q.reservation, 4), round(q.spread, 4), round(q.bid, 4), round(q.ask, 4)
(99.2, 1.6908, 98.3546, 100.0454)

Units: prices in currency, time in seconds (or any one unit used for every
parameter), ``sigma`` in price per sqrt(time), ``k`` per price unit, ``A``
fills per unit time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal, expm


@dataclass(frozen=True, slots=True)
class Quote:
    """Model quotes in price units. ``bid``/``ask`` are None when that side is pulled."""

    mid: float
    reservation: float
    bid: float | None
    ask: float | None

    @property
    def spread(self) -> float:
        if self.bid is None or self.ask is None:
            return math.nan
        return self.ask - self.bid


@dataclass(frozen=True, slots=True)
class ASParams:
    gamma: float  # CARA risk aversion, > 0
    sigma: float  # mid volatility, price per sqrt(time)
    k: float  # intensity decay per price unit, > 0
    q_max: int | None = None  # inventory bound: no bid at +q_max, no ask at -q_max

    def __post_init__(self) -> None:
        if not (self.gamma > 0 and self.sigma >= 0 and self.k > 0):
            raise ValueError("need gamma > 0, sigma >= 0, k > 0")
        if self.q_max is not None and self.q_max < 1:
            raise ValueError("q_max must be >= 1")


def _half_spread_floor(gamma: float, k: float) -> float:
    """``(1/gamma) ln(1 + gamma/k)``: tends to ``1/k`` as gamma -> 0."""
    return math.log1p(gamma / k) / gamma


def avellaneda_stoikov(mid: float, q: int, tau: float, p: ASParams) -> Quote:
    """Avellaneda-Stoikov (2008) quotes with ``tau = T - t`` time left."""
    if tau < 0:
        raise ValueError("tau must be >= 0")
    var = p.gamma * p.sigma**2 * tau
    r = mid - q * var
    half = 0.5 * var + _half_spread_floor(p.gamma, p.k)
    bid: float | None = r - half
    ask: float | None = r + half
    if p.q_max is not None:
        if q >= p.q_max:
            bid = None
        if q <= -p.q_max:
            ask = None
    return Quote(mid, r, bid, ask)


@dataclass(frozen=True, slots=True)
class GLFTParams:
    gamma: float
    sigma: float
    k: float
    A: float  # fill intensity at zero distance, per unit time
    q_max: int  # inventory bound Q

    def __post_init__(self) -> None:
        if not (self.gamma > 0 and self.sigma >= 0 and self.k > 0 and self.A > 0):
            raise ValueError("need gamma > 0, sigma >= 0, k > 0, A > 0")
        if self.q_max < 1:
            raise ValueError("q_max must be >= 1")

    @property
    def alpha(self) -> float:
        return 0.5 * self.k * self.gamma * self.sigma**2

    @property
    def eta(self) -> float:
        return self.A * (1.0 + self.gamma / self.k) ** (-(1.0 + self.k / self.gamma))


class GLFT:
    """Exact finite-horizon GLFT quotes plus the stationary and asymptotic forms.

    ``v`` is computed in log space-safe form: ``exp(-M tau) 1`` is normalised by
    its largest entry, which leaves every ratio ``v_q / v_{q+1}`` unchanged.
    """

    def __init__(self, p: GLFTParams) -> None:
        self.p = p
        Q = p.q_max
        self.qs = np.arange(-Q, Q + 1)
        diag = p.alpha * self.qs.astype(float) ** 2
        off = np.full(2 * Q, -p.eta)
        self._M = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        w, V = eigh_tridiagonal(diag, off)
        g = V[:, 0]
        self._ground = np.abs(g)  # Perron vector of -M: one sign, take it positive
        self._lambda0 = float(w[0])

    def _v(self, tau: float | None) -> np.ndarray:
        if tau is None:
            return self._ground
        if tau < 0:
            raise ValueError("tau must be >= 0")
        # Shift by the ground eigenvalue so exp() stays O(1) for long horizons.
        n = self._M.shape[0]
        v = expm(-(self._M - self._lambda0 * np.eye(n)) * tau) @ np.ones(n)
        return v / v.max()

    def deltas(self, q: int, tau: float | None) -> tuple[float | None, float | None]:
        """``(delta_b, delta_a)`` at inventory ``q``; ``tau=None`` is the stationary limit."""
        p = self.p
        Q = p.q_max
        if not -Q <= q <= Q:
            raise ValueError(f"q={q} outside [-{Q}, {Q}]")
        v = self._v(tau)
        i = q + Q
        base = _half_spread_floor(p.gamma, p.k)
        db = None if q == Q else math.log(v[i] / v[i + 1]) / p.k + base
        da = None if q == -Q else math.log(v[i] / v[i - 1]) / p.k + base
        return db, da

    def quote(self, mid: float, q: int, tau: float | None = None) -> Quote:
        db, da = self.deltas(q, tau)
        bid = None if db is None else mid - db
        ask = None if da is None else mid + da
        r = mid if db is None or da is None else mid + 0.5 * (da - db)
        return Quote(mid, r, bid, ask)


def glft_asymptotic(mid: float, q: int, p: GLFTParams) -> Quote:
    """The closed-form asymptotic GLFT quotes (GLFT 2013, section 4)."""
    base = _half_spread_floor(p.gamma, p.k)
    c = math.sqrt(p.sigma**2 * p.gamma / (2.0 * p.k * p.A)
                  * (1.0 + p.gamma / p.k) ** (1.0 + p.k / p.gamma))
    db = base + (2 * q + 1) / 2.0 * c
    da = base - (2 * q - 1) / 2.0 * c
    bid: float | None = mid - db
    ask: float | None = mid + da
    if q >= p.q_max:
        bid = None
    if q <= -p.q_max:
        ask = None
    return Quote(mid, mid + 0.5 * (da - db), bid, ask)
