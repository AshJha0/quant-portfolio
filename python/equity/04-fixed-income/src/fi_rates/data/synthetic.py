"""Seeded synthetic market data: deposit/swap quote sets and a bond portfolio.

Everything here is deterministic given ``seed`` and fully offline — this is
the only data source the tests and the example pipeline use.

Curve variants
--------------
``upward``    realistic post-2023-normalisation upward-sloping curve
              (short ~3.0%, 30y ~4.6%).
``inverted``  2022/23-style inversion (short ~5.3%, 10y ~4.0%).
``flat``      everything ~3.5%.
``negative``  2019-Europe-style: negative out to ~7y, mildly positive long end.

Quotes are deposits at 0.25/0.5/1y (simple rates) and annual-fixed-leg par
swaps at 2..30y.  A small seeded idiosyncratic perturbation (a few tenths of
a bp) makes the quotes look like real closes without breaking monotonicity.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from ..bond import FixedRateBond
from ..bootstrap import Deposit, Instrument, ParSwap
from ..risk import Position

__all__ = ["market_quotes", "sample_portfolio", "CURVE_VARIANTS"]

CURVE_VARIANTS: tuple[str, ...] = ("upward", "inverted", "flat", "negative")

_DEPO_TENORS = (0.25, 0.5, 1.0)
_SWAP_TENORS = (2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)

# (short_rate, long_rate, decay) for a Nelson-Siegel-ish level/slope shape:
# r(t) = long + (short - long) * (1 - exp(-t/decay)) / (t/decay)  ... roughly.
_VARIANT_PARAMS: dict[str, tuple[float, float, float]] = {
    "upward": (0.030, 0.047, 3.0),
    "inverted": (0.0533, 0.0392, 4.0),
    "flat": (0.035, 0.035, 5.0),
    "negative": (-0.0060, 0.0075, 6.0),
}


def _base_rate(t: np.ndarray, short: float, long_: float, decay: float) -> np.ndarray:
    x = t / decay
    slope_loading = (1.0 - np.exp(-x)) / np.where(x > 1e-12, x, 1e-12)
    return long_ + (short - long_) * slope_loading


def market_quotes(
    variant: str = "upward", seed: int = 42, noise_bp: float = 0.3
) -> list[Instrument]:
    """Deposit + par swap quote set for a curve variant.

    Parameters
    ----------
    variant : str
        One of :data:`CURVE_VARIANTS`.
    seed : int
        Seed for the tiny idiosyncratic quote perturbation.
    noise_bp : float
        Std-dev of the perturbation in basis points (default 0.3bp; set 0 for
        exact smooth quotes).

    Returns
    -------
    list of :class:`~fi_rates.bootstrap.Deposit` and
    :class:`~fi_rates.bootstrap.ParSwap` sorted by maturity.
    """
    if variant not in CURVE_VARIANTS:
        raise ValueError(
            f"unknown curve variant {variant!r}; choose from {CURVE_VARIANTS}"
        )
    short, long_, decay = _VARIANT_PARAMS[variant]
    rng = np.random.default_rng(seed)
    quotes: list[Instrument] = []
    for t in _DEPO_TENORS:
        r = float(_base_rate(np.array([t]), short, long_, decay)[0])
        r += float(rng.normal(0.0, noise_bp * 1e-4))
        quotes.append(Deposit(maturity=t, rate=r))
    for t in _SWAP_TENORS:
        r = float(_base_rate(np.array([t]), short, long_, decay)[0])
        r += float(rng.normal(0.0, noise_bp * 1e-4))
        quotes.append(ParSwap(maturity=t, rate=r, frequency=1))
    return quotes


def sample_portfolio(settlement: dt.date, seed: int = 42) -> list[Position]:
    """Sample government + corporate bond portfolio.

    Governments carry zero z-spread; corporates carry seeded z-spreads in
    roughly the 80-250bp range (spread risk handled as a z-spread over the
    single discount curve — see METHODOLOGY assumptions register).

    Quantities are in bonds (face 100 each).
    """
    rng = np.random.default_rng(seed)

    def _mk(years: float, coupon: float, freq: int, dcc: str) -> FixedRateBond:
        maturity = settlement + dt.timedelta(days=int(round(years * 365.25)))
        effective = settlement - dt.timedelta(days=200)  # seasoned bond
        return FixedRateBond(
            effective=effective,
            maturity=maturity,
            coupon=coupon,
            frequency=freq,
            daycount=dcc,
        )

    govts = [
        ("GOVT_2Y", _mk(2.0, 0.0325, 2, "ACT/ACT-ISDA"), 5000.0),
        ("GOVT_5Y", _mk(5.0, 0.0350, 2, "ACT/ACT-ISDA"), 4000.0),
        ("GOVT_10Y", _mk(10.0, 0.0400, 2, "ACT/ACT-ISDA"), 3000.0),
        ("GOVT_30Y", _mk(29.8, 0.0450, 2, "ACT/ACT-ISDA"), 1500.0),
    ]
    corps = [
        ("CORP_A_3Y", _mk(3.1, 0.0475, 2, "30/360US")),
        ("CORP_BBB_7Y", _mk(7.2, 0.0550, 2, "30/360US")),
        ("CORP_BBB_12Y", _mk(12.4, 0.0585, 2, "30/360US")),
    ]
    positions = [
        Position(bond=b, quantity=q, z_spread=0.0, label=lbl) for lbl, b, q in govts
    ]
    for lbl, b in corps:
        spread = float(rng.uniform(0.008, 0.025))  # 80-250bp
        qty = float(rng.integers(500, 2500))
        positions.append(Position(bond=b, quantity=qty, z_spread=spread, label=lbl))
    return positions
