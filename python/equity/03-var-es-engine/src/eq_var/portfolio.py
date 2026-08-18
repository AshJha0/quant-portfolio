"""Portfolio representation and P&L revaluation.

Conventions
-----------
* Risk factors are either *price* factors (stock prices, index levels) or
  *vol* factors (implied volatilities).
* A **scenario** is a vector of factor moves, one entry per risk factor, in the
  order of ``Portfolio.factor_names``:

  - price factors: simple (arithmetic) return, e.g. ``-0.05`` = -5 %;
  - vol factors: *absolute* change in implied vol, e.g. ``+0.10`` = +10 vol pts.

* P&L is in currency units (dollars); losses are negative P&L.  VaR/ES
  functions elsewhere report positive numbers for losses.
* Options are European, priced with Black-Scholes (ACT/365F, continuously
  compounded ``r`` and dividend yield ``q``, annualised log-return vol).
  Horizon P&L keeps time-to-expiry fixed (no theta bleed) so that the
  delta-gamma-vega approximation and full revaluation measure the same thing;
  this is documented as an assumption in docs/METHODOLOGY.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy.stats import norm

__all__ = [
    "RiskFactor",
    "Position",
    "EquityPosition",
    "FuturePosition",
    "OptionPosition",
    "Portfolio",
    "bs_price",
    "bs_greeks",
]

FactorKind = Literal["equity", "index", "vol"]


@dataclass(frozen=True)
class RiskFactor:
    """A market risk factor.

    Parameters
    ----------
    name : str
        Unique identifier, e.g. ``"AAPL"``, ``"SPX"``, ``"SPX_IV"``.
    kind : {"equity", "index", "vol"}
        Factor type. ``"vol"`` factors move in absolute vol points.
    level : float
        Current level (price for price factors, decimal vol for vol factors).
    """

    name: str
    kind: FactorKind
    level: float

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError(f"factor {self.name!r}: level must be >= 0, got {self.level}")


# --------------------------------------------------------------------------- #
# Black-Scholes helpers (self-contained, used for option revaluation)
# --------------------------------------------------------------------------- #
def bs_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    div_yield: float,
    vol: float,
    kind: Literal["call", "put"],
) -> float:
    """Black-Scholes European option price.

    Units: ``rate``/``div_yield`` continuously compounded annualised, ``tau``
    in years (ACT/365F), ``vol`` annualised log-return vol.  Handles the
    ``tau -> 0`` and ``vol -> 0`` limits via intrinsic/forward-intrinsic value.
    """
    if spot < 0 or strike < 0:
        raise ValueError("spot and strike must be non-negative")
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if vol < 0:
        raise ValueError(f"vol must be >= 0, got {vol}")
    sign = 1.0 if kind == "call" else -1.0
    if tau == 0.0:
        return max(sign * (spot - strike), 0.0)
    if vol == 0.0 or spot == 0.0 or strike == 0.0:
        fwd = spot * np.exp((rate - div_yield) * tau)
        return float(np.exp(-rate * tau) * max(sign * (fwd - strike), 0.0))
    sq = vol * np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * tau) / sq
    d2 = d1 - sq
    return float(
        sign
        * (
            spot * np.exp(-div_yield * tau) * norm.cdf(sign * d1)
            - strike * np.exp(-rate * tau) * norm.cdf(sign * d2)
        )
    )


def bs_greeks(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    div_yield: float,
    vol: float,
    kind: Literal["call", "put"],
) -> dict[str, float]:
    """Black-Scholes delta, gamma and vega (per 1.00 of vol, i.e. per 100 pts).

    Returns
    -------
    dict with keys ``delta`` (dV/dS), ``gamma`` (d2V/dS2) and ``vega``
    (dV/dsigma for sigma in decimals).
    """
    if tau <= 0 or vol <= 0 or spot <= 0:
        # Degenerate: delta is a step function, gamma/vega collapse to 0.
        intrinsic_delta = 0.0
        if tau <= 0:
            if kind == "call":
                intrinsic_delta = 1.0 if spot > strike else 0.0
            else:
                intrinsic_delta = -1.0 if spot < strike else 0.0
        return {"delta": intrinsic_delta, "gamma": 0.0, "vega": 0.0}
    sq = vol * np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * tau) / sq
    disc_q = np.exp(-div_yield * tau)
    delta = disc_q * norm.cdf(d1) if kind == "call" else -disc_q * norm.cdf(-d1)
    gamma = disc_q * norm.pdf(d1) / (spot * sq)
    vega = spot * disc_q * norm.pdf(d1) * np.sqrt(tau)
    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


# --------------------------------------------------------------------------- #
# Positions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Position:
    """Base class for positions.  Subclasses implement the P&L mapping."""

    name: str

    def factor_names(self) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def pnl(
        self,
        factors: dict[str, RiskFactor],
        shocks: dict[str, np.ndarray],
        method: str,
    ) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class EquityPosition(Position):
    """Cash equity position: ``shares`` of ``factor`` (long > 0, short < 0)."""

    factor: str = ""
    shares: float = 0.0

    def factor_names(self) -> list[str]:
        return [self.factor]

    def pnl(self, factors, shocks, method) -> np.ndarray:
        s0 = factors[self.factor].level
        return self.shares * s0 * shocks[self.factor]


@dataclass(frozen=True)
class FuturePosition(Position):
    """Equity index future: linear in the index level.

    P&L = contracts * multiplier * F0 * return.  Daily margining makes futures
    P&L linear in the futures price; we shock the index level directly and
    ignore the (deterministic, small over 1-10d) basis carry.
    """

    factor: str = ""
    contracts: float = 0.0
    multiplier: float = 50.0

    def factor_names(self) -> list[str]:
        return [self.factor]

    def pnl(self, factors, shocks, method) -> np.ndarray:
        f0 = factors[self.factor].level
        return self.contracts * self.multiplier * f0 * shocks[self.factor]


@dataclass(frozen=True)
class OptionPosition(Position):
    """European option on a price factor with an implied-vol factor.

    Parameters
    ----------
    underlier, vol_factor : str
        Names of the price and implied-vol risk factors.
    strike, expiry : float
        Strike and time-to-expiry in years (ACT/365F).
    rate, div_yield : float
        Continuously compounded annualised rate / dividend yield.
    kind : {"call", "put"}
    contracts, multiplier : float
        Signed number of contracts (short < 0) and contract multiplier.
    """

    underlier: str = ""
    vol_factor: str = ""
    strike: float = 100.0
    expiry: float = 0.25
    rate: float = 0.0
    div_yield: float = 0.0
    kind: Literal["call", "put"] = "call"
    contracts: float = 0.0
    multiplier: float = 100.0

    def factor_names(self) -> list[str]:
        return [self.underlier, self.vol_factor]

    def _scale(self) -> float:
        return self.contracts * self.multiplier

    def price(self, spot: float, vol: float) -> float:
        """Unit option price at given spot / implied vol."""
        return bs_price(spot, self.strike, self.expiry, self.rate, self.div_yield, vol, self.kind)

    def greeks(self, spot: float, vol: float) -> dict[str, float]:
        return bs_greeks(spot, self.strike, self.expiry, self.rate, self.div_yield, vol, self.kind)

    def pnl(self, factors, shocks, method) -> np.ndarray:
        s0 = factors[self.underlier].level
        v0 = factors[self.vol_factor].level
        ds = s0 * np.asarray(shocks[self.underlier], dtype=float)
        dv = np.asarray(shocks[self.vol_factor], dtype=float)
        if method == "delta_gamma":
            g = self.greeks(s0, v0)
            unit = g["delta"] * ds + 0.5 * g["gamma"] * ds**2 + g["vega"] * dv
            return self._scale() * unit
        # full revaluation
        p0 = self.price(s0, v0)
        spots = s0 + ds
        vols = np.maximum(v0 + dv, 0.0)  # floor implied vol at zero
        prices = np.array([self.price(s, v) for s, v in zip(spots, vols)])
        return self._scale() * (prices - p0)


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
@dataclass
class Portfolio:
    """A portfolio of positions mapped onto a common set of risk factors.

    Parameters
    ----------
    positions : list of Position
    factors : dict mapping factor name -> RiskFactor
        Must contain every factor referenced by a position.
    """

    positions: list[Position] = field(default_factory=list)
    factors: dict[str, RiskFactor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [
            f for p in self.positions for f in p.factor_names() if f not in self.factors
        ]
        if missing:
            raise ValueError(f"positions reference unknown risk factors: {sorted(set(missing))}")

    # -- structure ---------------------------------------------------------- #
    @property
    def factor_names(self) -> list[str]:
        """Risk factor names in canonical (insertion) order."""
        return list(self.factors.keys())

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    def value(self) -> float:
        """Current mark-to-market value (futures contribute 0, options premium)."""
        total = 0.0
        for p in self.positions:
            if isinstance(p, EquityPosition):
                total += p.shares * self.factors[p.factor].level
            elif isinstance(p, OptionPosition):
                s0 = self.factors[p.underlier].level
                v0 = self.factors[p.vol_factor].level
                total += p._scale() * p.price(s0, v0)
            # futures: zero present value at inception of the margin period
        return total

    # -- P&L revaluation ---------------------------------------------------- #
    def pnl(
        self,
        scenarios: np.ndarray | Sequence[Sequence[float]],
        method: Literal["full", "delta_gamma"] = "full",
    ) -> np.ndarray:
        """Revalue the portfolio under factor-move scenarios.

        Parameters
        ----------
        scenarios : array (n_scenarios, n_factors) or (n_factors,)
            Factor moves in ``factor_names`` order (returns for price factors,
            absolute vol changes for vol factors).
        method : {"full", "delta_gamma"}
            ``"full"`` fully reprices options; ``"delta_gamma"`` uses the
            delta-gamma-vega Taylor approximation.  Linear positions are exact
            under both.

        Returns
        -------
        ndarray (n_scenarios,) of P&L in currency units (loss < 0).
        """
        if method not in ("full", "delta_gamma"):
            raise ValueError(f"method must be 'full' or 'delta_gamma', got {method!r}")
        scen = np.atleast_2d(np.asarray(scenarios, dtype=float))
        if scen.shape[1] != self.n_factors:
            raise ValueError(
                f"scenarios have {scen.shape[1]} columns, portfolio has "
                f"{self.n_factors} factors ({self.factor_names})"
            )
        shocks = {name: scen[:, j] for j, name in enumerate(self.factor_names)}
        total = np.zeros(scen.shape[0])
        for p in self.positions:
            total += p.pnl(self.factors, shocks, method)
        return total

    def approximation_error(self, scenarios: np.ndarray) -> np.ndarray:
        """Delta-gamma P&L minus full-revaluation P&L per scenario."""
        return self.pnl(scenarios, "delta_gamma") - self.pnl(scenarios, "full")

    # -- linear/quadratic risk mapping -------------------------------------- #
    def delta_exposures(self) -> np.ndarray:
        """Dollar sensitivities to a unit factor move, in ``factor_names`` order.

        Price factor entry = dollar delta (dP&L per unit *return*); vol factor
        entry = dollar vega (dP&L per unit *absolute* vol change).  This is the
        exposure vector ``w`` used by parametric VaR: sigma_p^2 = w' Sigma w.
        """
        expo = dict.fromkeys(self.factor_names, 0.0)
        for p in self.positions:
            if isinstance(p, EquityPosition):
                expo[p.factor] += p.shares * self.factors[p.factor].level
            elif isinstance(p, FuturePosition):
                expo[p.factor] += p.contracts * p.multiplier * self.factors[p.factor].level
            elif isinstance(p, OptionPosition):
                s0 = self.factors[p.underlier].level
                v0 = self.factors[p.vol_factor].level
                g = p.greeks(s0, v0)
                expo[p.underlier] += p._scale() * g["delta"] * s0
                expo[p.vol_factor] += p._scale() * g["vega"]
        return np.array([expo[f] for f in self.factor_names])

    def gamma_matrix(self) -> np.ndarray:
        """Dollar gamma matrix G: quadratic P&L term is 0.5 * x' G x.

        ``x`` is the factor-move vector.  Options contribute
        ``contracts * mult * gamma * S0^2`` on the underlier diagonal (since
        dS = S0 * x); cross terms are zero for vanilla BS positions.
        """
        n = self.n_factors
        idx = {f: j for j, f in enumerate(self.factor_names)}
        gmat = np.zeros((n, n))
        for p in self.positions:
            if isinstance(p, OptionPosition):
                s0 = self.factors[p.underlier].level
                v0 = self.factors[p.vol_factor].level
                g = p.greeks(s0, v0)
                j = idx[p.underlier]
                gmat[j, j] += p._scale() * g["gamma"] * s0**2
        return gmat
