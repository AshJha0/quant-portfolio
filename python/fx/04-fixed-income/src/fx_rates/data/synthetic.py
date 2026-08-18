"""Seeded synthetic two-currency market quote sets and sample position books.

Quotes are generated from analytic Nelson–Siegel "true" zero curves, so the
bootstrap can be validated by an exact round trip: par quotes are computed
*from* the true discount factors, hence bootstrapping the quotes must recover
the true DFs at every pillar to machine precision.

Regimes
-------
- ``normal``       : upward-sloping USD (≈3.5→4.5%) above EUR (≈2.2→2.9%),
                     basis ≈ -12…-30 bp (post-2015 average conditions).
- ``inverted``     : late-cycle inverted USD curve (≈5.4→4.1%).
- ``negative_eur`` : 2019-style — EUR front end at ≈-0.5% (negative deposit
                     rates), USD ≈2%, basis ≈-15…-40 bp.
- ``crisis``       : 2008-style wide basis: front-end basis ≈-180 bp
                     decaying to ≈-55 bp at 10y.

Every generator takes an explicit ``seed``; noise is applied to the *curve
parameters* (a few bp), never to individual quotes, so internal consistency
(and the bootstrap round trip) is preserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bootstrap import bootstrap_curve, par_swap_rate
from ..curve import DiscountCurve
from ..fxforward import FXForward, FXSwap, MarketState, market_forward
from ..xccy import CrossCurrencySwap, solve_par_rate_quote

__all__ = [
    "REGIMES",
    "SyntheticMarket",
    "generate_market_quotes",
    "build_market_state",
    "sample_book",
    "third_currency_curve",
]

REGIMES = ("normal", "inverted", "negative_eur", "crisis")

# A complete annual par-swap strip (2..10y).  Real markets quote a sparse set
# (2,3,4,5,7,10) and interpolate the missing par rates — using the complete
# strip here keeps the bootstrap round trip against the true generator curve
# exact at every pillar, which is what the validation suite asserts.
DEPOSIT_TENORS = (0.25, 0.5, 1.0)
SWAP_TENORS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
FORWARD_TENORS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)

# Nelson–Siegel parameters (beta0, beta1, beta2, tau_ns) per regime/ccy,
# and basis spread pillars (T, bp).
_PARAMS = {
    "normal": {
        "USD": (0.045, -0.010, -0.004, 1.8),
        "EUR": (0.029, -0.007, -0.003, 1.8),
        "spot": 1.0850,
        "basis_bp": ((0.25, -12.0), (1.0, -15.0), (2.0, -18.0), (3.0, -20.0),
                     (5.0, -25.0), (7.0, -28.0), (10.0, -30.0)),
    },
    "inverted": {
        "USD": (0.041, 0.013, 0.004, 1.5),
        "EUR": (0.031, 0.008, 0.002, 1.5),
        "spot": 1.0550,
        "basis_bp": ((0.25, -20.0), (1.0, -22.0), (2.0, -24.0), (3.0, -25.0),
                     (5.0, -27.0), (7.0, -28.0), (10.0, -30.0)),
    },
    "negative_eur": {
        "USD": (0.019, 0.005, 0.002, 1.5),
        "EUR": (0.004, -0.009, -0.002, 1.8),
        "spot": 1.1200,
        "basis_bp": ((0.25, -15.0), (1.0, -20.0), (2.0, -25.0), (3.0, -28.0),
                     (5.0, -33.0), (7.0, -37.0), (10.0, -40.0)),
    },
    "crisis": {
        "USD": (0.030, -0.020, -0.010, 1.2),
        "EUR": (0.035, -0.005, -0.005, 1.2),
        "spot": 1.3400,
        "basis_bp": ((0.25, -180.0), (0.5, -160.0), (1.0, -140.0), (2.0, -110.0),
                     (3.0, -90.0), (5.0, -70.0), (7.0, -60.0), (10.0, -55.0)),
    },
}


def _nelson_siegel(t, beta0, beta1, beta2, tau_ns):
    x = np.asarray(t, dtype=float) / tau_ns
    with np.errstate(divide="ignore", invalid="ignore"):
        loading = np.where(x > 0, (1.0 - np.exp(-x)) / np.where(x > 0, x, 1.0), 1.0)
    return beta0 + beta1 * loading + beta2 * (loading - np.exp(-x))


@dataclass(frozen=True)
class SyntheticMarket:
    """A complete, internally consistent two-currency quote set.

    ``true_domestic`` / ``true_foreign`` are the exact generator curves used
    only for round-trip validation; production code sees only the quotes.
    Basis spreads are decimal; FX forward quotes are basis-consistent:
    ``F_mkt(T) = spot * DF_f(T) * exp(-s(T) T) / DF_d(T)``.
    """

    regime: str
    seed: int
    spot: float
    domestic_deposits: tuple[tuple[float, float], ...]
    domestic_swaps: tuple[tuple[float, float], ...]
    foreign_deposits: tuple[tuple[float, float], ...]
    foreign_swaps: tuple[tuple[float, float], ...]
    basis_spreads: tuple[tuple[float, float], ...]
    fx_forward_quotes: tuple[tuple[float, float], ...]
    true_domestic: DiscountCurve
    true_foreign: DiscountCurve
    pair: tuple[str, str] = ("EUR", "USD")


def _true_curve(params, noise: float, name: str) -> DiscountCurve:
    b0, b1, b2, tau_ns = params
    times = np.unique(np.concatenate([np.asarray(DEPOSIT_TENORS),
                                      np.arange(1.0, 10.5, 0.25)]))
    zeros = _nelson_siegel(times, b0 + noise, b1, b2, tau_ns)
    return DiscountCurve.from_zero_rates(times, zeros, name=name)


def _quotes_from_curve(curve: DiscountCurve):
    deposits = tuple(
        (t, (1.0 / curve.df(t) - 1.0) / t) for t in DEPOSIT_TENORS
    )
    swaps = tuple((float(n), par_swap_rate(curve, n)) for n in SWAP_TENORS)
    return deposits, swaps


def generate_market_quotes(regime: str = "normal", seed: int = 0) -> SyntheticMarket:
    """Generate a seeded, internally consistent EURUSD quote set.

    Raises
    ------
    ValueError
        For an unknown regime.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    rng = np.random.default_rng(seed)
    p = _PARAMS[regime]
    # seed-dependent level noise of a few bp on each curve, and a small
    # spot perturbation; quotes stay exactly consistent with the curves.
    usd_noise, eur_noise = rng.normal(0.0, 3e-4, size=2)
    spot = float(p["spot"] * (1.0 + rng.normal(0.0, 0.002)))
    true_usd = _true_curve(p["USD"], usd_noise, "USD-true")
    true_eur = _true_curve(p["EUR"], eur_noise, "EUR-true")
    usd_deps, usd_swps = _quotes_from_curve(true_usd)
    eur_deps, eur_swps = _quotes_from_curve(true_eur)
    basis = tuple((t, bp * 1e-4) for t, bp in p["basis_bp"])
    bt = np.asarray([t for t, _ in basis])
    bv = np.asarray([s for _, s in basis])
    forwards = tuple(
        (
            t,
            float(
                spot * true_eur.df(t)
                * np.exp(-np.interp(t, bt, bv) * t)
                / true_usd.df(t)
            ),
        )
        for t in FORWARD_TENORS
    )
    return SyntheticMarket(
        regime=regime,
        seed=seed,
        spot=spot,
        domestic_deposits=usd_deps,
        domestic_swaps=usd_swps,
        foreign_deposits=eur_deps,
        foreign_swaps=eur_swps,
        basis_spreads=basis,
        fx_forward_quotes=forwards,
        true_domestic=true_usd,
        true_foreign=true_eur,
    )


def build_market_state(quotes: SyntheticMarket) -> MarketState:
    """Bootstrap both curves from the quotes and assemble a ``MarketState``."""
    dom = bootstrap_curve(quotes.domestic_deposits, quotes.domestic_swaps,
                          name=quotes.pair[1])
    for_ = bootstrap_curve(quotes.foreign_deposits, quotes.foreign_swaps,
                           name=quotes.pair[0])
    return MarketState(
        spot=quotes.spot,
        domestic_curve=dom,
        foreign_curve=for_,
        basis_spreads=quotes.basis_spreads,
        pair=quotes.pair,
    )


def sample_book(market: MarketState, seed: int = 0) -> list:
    """A small representative desk book: two outrights, an FX swap and a
    5y cross-currency swap, struck near (but not at) current market levels
    so every position carries P&L and risk.
    """
    rng = np.random.default_rng(seed)
    off = rng.normal(0.0, 0.004, size=3)  # strike offsets vs market forward
    f6m = market_forward(market, 0.5)
    f2y = market_forward(market, 2.0)
    f3m = market_forward(market, 0.25)
    f1y = market_forward(market, 1.0)
    book: list = [
        FXForward(25e6, f6m * (1.0 + off[0]), 0.5, market.pair,
                  label="Long 25m 6m outright"),
        FXForward(-15e6, f2y * (1.0 + off[1]), 2.0, market.pair,
                  label="Short 15m 2y outright"),
        FXSwap(40e6, f3m, 0.25, f1y * (1.0 + off[2]), 1.0, market.pair,
               label="40m 3m/1y buy-sell FX swap"),
    ]
    xccy = CrossCurrencySwap(
        notional_base=50e6,
        notional_quote=50e6 * market.spot,
        rate_base=max(market.foreign_curve.zero_rate(5.0), 0.0) + 0.002,
        rate_quote=0.0,  # solved to par below, then perturbed
        maturity=5.0,
        frequency=1,
        receive_base=True,
        pair=market.pair,
        label="50m 5y xccy rec EUR",
    )
    par_q = solve_par_rate_quote(xccy, market)
    from dataclasses import replace
    book.append(replace(xccy, rate_quote=par_q + 0.0015))
    return book


def third_currency_curve(seed: int = 0) -> DiscountCurve:
    """A JPY-style low-rate curve for triangular-consistency tests."""
    rng = np.random.default_rng(seed)
    times = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    zeros = _nelson_siegel(times, 0.008 + rng.normal(0.0, 2e-4), -0.006, 0.002, 2.0)
    return DiscountCurve.from_zero_rates(times, zeros, name="JPY")
