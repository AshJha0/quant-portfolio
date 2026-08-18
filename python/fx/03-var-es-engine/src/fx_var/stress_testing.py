"""FX stress testing: historical replays, hypothetical scenarios, peg breaks,
sensitivity ladders and reverse stress.

Stress is the complement to VaR, not a substitute: HS and var-covar are
*blind* to risks absent from the estimation window (a pegged currency has
no history of breaking - until it does).  Every scenario here is a factor
shock dictionary applied through full revaluation, so options reprice
convexly and forwards feel their rate legs.

Historical replay calibrations (one-day moves vs USD unless stated,
documented sources in docs/METHODOLOGY.md):

* **GBP flash - Brexit referendum, 24 Jun 2016**: GBPUSD -8.1% (1.4877 ->
  1.3679), EURUSD -2.4%, JPY +3.9% (safe haven), 1m GBP vols +12 pts.
* **CHF depeg - SNB floor removal, 15 Jan 2015**: CHFUSD +14.9% close-to-
  close (intraday +30%+), EURUSD -1.4%, CHF vols +15 pts.  The canonical
  peg-break: the *prior 250 days* of USDCHF had no daily move over 1.9%.
* **JPY carry unwind, 7-8 Oct 1998**: JPYUSD +11.5% over two days as
  USDJPY fell 131 -> 117 (LTCM deleveraging); AUD -4%.
* **EM crisis composite** (1997 THB / 1998 RUB / 2001 ARS-shaped): EM
  currencies -12% to -25%, EM vols +10-20 pts, G10 risk-off (JPY up).

Reverse stress: for a linearised book with exposures ``w`` and factor
covariance ``Sigma``, the most damaging shock at Mahalanobis radius k is
``dx* = -k Sigma w / sqrt(w' Sigma w)`` with loss ``k sqrt(w' Sigma w)``
- closed form, verified numerically in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .book import Book, Market
from .common import fx_factor, vol_factor

__all__ = [
    "Scenario",
    "historical_scenarios",
    "usd_broad_move",
    "peg_break_scenario",
    "run_stress",
    "sensitivity_ladder",
    "reverse_stress_linear",
    "reverse_stress_numerical",
]


@dataclass(frozen=True)
class Scenario:
    """A named stress scenario: factor shocks + description.

    ``shocks`` maps factor names to shocks in engine units (``FX:*`` log
    returns, ``IR:*``/``VOL:*`` absolute).  Shocks for factors the book does
    not carry are ignored at run time, so one scenario library serves every
    book.
    """

    name: str
    shocks: Mapping[str, float]
    description: str = ""


def _ln(pct: float) -> float:
    """Convert a simple percentage move to a log return."""
    return float(np.log1p(pct))


def historical_scenarios() -> dict[str, Scenario]:
    """Library of calibrated historical FX replay scenarios.

    Shock sizes are close-to-close calibrations of the episodes in the
    module docstring, expressed on USD factors (``FX:CCY`` = log return of
    CCYUSD) plus implied-vol add-ons for the pairs that gapped.
    """
    lib = {
        "brexit_2016": Scenario(
            "GBP flash - Brexit referendum (24 Jun 2016)",
            {
                fx_factor("GBP"): _ln(-0.081),
                fx_factor("EUR"): _ln(-0.024),
                fx_factor("JPY"): _ln(+0.039),
                fx_factor("CHF"): _ln(+0.015),
                fx_factor("AUD"): _ln(-0.019),
                vol_factor("GBPUSD"): 0.12,
                vol_factor("EURUSD"): 0.04,
            },
            "Cable -8.1% in a day; safe havens bid; GBP vol +12 pts.",
        ),
        "chf_depeg_2015": Scenario(
            "CHF depeg - SNB floor removal (15 Jan 2015)",
            {
                fx_factor("CHF"): _ln(+0.149),
                fx_factor("EUR"): _ln(-0.014),
                fx_factor("JPY"): _ln(+0.012),
                vol_factor("USDCHF"): 0.15,
                vol_factor("EURUSD"): 0.02,
            },
            "CHF +14.9% vs USD close-to-close (intraday >+30%); the "
            "peg-break archetype - invisible to a 250d HS window.",
        ),
        "jpy_1998": Scenario(
            "JPY carry unwind (7-8 Oct 1998)",
            {
                fx_factor("JPY"): _ln(+0.115),
                fx_factor("AUD"): _ln(-0.040),
                fx_factor("NZD"): _ln(-0.040),
                fx_factor("CHF"): _ln(+0.020),
                vol_factor("USDJPY"): 0.10,
            },
            "USDJPY 131->117 in two sessions as levered carry unwound.",
        ),
        "em_crisis": Scenario(
            "EM crisis composite (1997-2001 shaped)",
            {
                fx_factor("MXN"): _ln(-0.12),
                fx_factor("BRL"): _ln(-0.18),
                fx_factor("TRY"): _ln(-0.25),
                fx_factor("ZAR"): _ln(-0.15),
                fx_factor("JPY"): _ln(+0.05),
                fx_factor("AUD"): _ln(-0.05),
                vol_factor("USDMXN"): 0.10,
                vol_factor("USDBRL"): 0.15,
                vol_factor("USDTRY"): 0.20,
            },
            "Broad EM devaluation with G10 risk-off and EM vol explosion; "
            "correlations inside the EM block go to ~1.",
        ),
    }
    return lib


def usd_broad_move(ccys: Sequence[str], pct: float) -> Scenario:
    """Hypothetical broad USD move: USD strengthens by ``pct`` vs every ccy.

    ``pct=+0.10`` = USD +10% (every CCYUSD falls 10% in simple terms);
    ``pct=-0.10`` = USD -10%.  USD itself is skipped.
    """
    if pct <= -1.0:
        raise ValueError("pct must be > -100%")
    shocks = {fx_factor(c): _ln(-pct / (1.0 + pct)) for c in ccys if c.upper() != "USD"}
    label = f"USD {'+' if pct >= 0 else ''}{pct:.0%} broad move"
    return Scenario(label, shocks, "Uniform USD move against all book currencies.")


def peg_break_scenario(
    ccy: str,
    jump: float = -0.30,
    vol_spike: float = 0.15,
    vol_pairs: Sequence[str] = (),
    contagion: Mapping[str, float] | None = None,
) -> Scenario:
    """Peg-break stress add-on for a pegged/managed currency.

    This is the mandatory companion to any HS/parametric VaR on a book
    holding pegged currencies (the engine's PegBlindnessWarning points
    here): the scenario supplies the revaluation jump that the historical
    window cannot contain.

    Parameters
    ----------
    ccy : str
        The pegged currency, e.g. ``"HKD"``.
    jump : float
        Simple revaluation size vs USD: -0.30 = 30% devaluation; a
        positive value models a CHF-2015-style upward break.
    vol_spike : float
        Absolute implied-vol add-on applied to ``vol_pairs``.
    vol_pairs : sequence of str
        Option pairs whose vols spike (e.g. ``["USDHKD"]``).
    contagion : mapping ccy -> float, optional
        Simple-percentage co-moves for other currencies
        (e.g. ``{"SAR": -0.05}``).
    """
    if jump <= -1.0:
        raise ValueError("jump must be > -100%")
    shocks: dict[str, float] = {fx_factor(ccy): _ln(jump)}
    for p in vol_pairs:
        shocks[vol_factor(p)] = vol_spike
    for c, m in (contagion or {}).items():
        shocks[fx_factor(c)] = _ln(m)
    direction = "devaluation" if jump < 0 else "revaluation"
    return Scenario(
        f"{ccy.upper()} peg break ({jump:+.0%} {direction})",
        shocks,
        f"Managed-currency regime break: {ccy.upper()} gaps {jump:+.0%} vs "
        "USD with no intermediate prints; HS/parametric VaR see none of it.",
    )


def run_stress(
    book: Book,
    market: Market,
    scenarios: Mapping[str, Scenario] | Sequence[Scenario],
    option_method: str = "full",
) -> pd.DataFrame:
    """Full-revaluation P&L of ``book`` under each scenario.

    Shocks for factors the book does not carry are dropped (with the book's
    factor list as the filter), so library scenarios apply to any book.

    Returns
    -------
    pandas.DataFrame
        Indexed by scenario key with columns ``pnl`` (base ccy) and
        ``description``, sorted worst-first.
    """
    if isinstance(scenarios, Mapping):
        items = list(scenarios.items())
    else:
        items = [(s.name, s) for s in scenarios]
    factors = set(book.factors(market))
    rows = {}
    for key, sc in items:
        shocks = {f: v for f, v in sc.shocks.items() if f in factors}
        pnl = float(book.pnl(market, shocks, option_method=option_method))
        rows[key] = (sc.name, pnl, sc.description)
    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=["scenario", "pnl", "description"])
    return out.sort_values("pnl")


def sensitivity_ladder(
    book: Book,
    market: Market,
    factor: str,
    shocks: Sequence[float] | None = None,
    option_method: str = "full",
) -> pd.DataFrame:
    """P&L ladder for one factor over a grid of shocks (full revaluation).

    Default grid for FX factors: -10% .. +10% in simple terms; for IR
    factors: -200bp .. +200bp; for VOL factors: -10 .. +10 vol pts.
    """
    if shocks is None:
        if factor.startswith("FX:"):
            grid = [_ln(p) for p in (-0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10)]
        elif factor.startswith("IR:"):
            grid = [-0.02, -0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01, 0.02]
        elif factor.startswith("VOL:"):
            grid = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
        else:
            raise ValueError(f"unknown factor family for {factor!r}")
    else:
        grid = [float(s) for s in shocks]
    pnls = [float(book.pnl(market, {factor: s}, option_method=option_method)) for s in grid]
    return pd.DataFrame({"shock": grid, "pnl": pnls})


def reverse_stress_linear(
    exposures: pd.Series, cov: pd.DataFrame, loss_target: float | None = None,
    radius: float | None = None,
) -> tuple[pd.Series, float]:
    """Closed-form reverse stress for a linear book.

    Among all factor shocks ``dx`` with Mahalanobis norm
    ``sqrt(dx' Sigma^{-1} dx) <= k``, the loss ``-w'dx`` is maximised at
    ``dx* = -k Sigma w / sqrt(w' Sigma w)`` with maximum loss
    ``k sqrt(w' Sigma w)``.  Specify either ``radius`` (k) or
    ``loss_target`` (solves for k).

    Returns
    -------
    (shocks, loss)
        The worst-case shock vector (Series over ``exposures.index``) and
        the loss it produces.
    """
    w = exposures.to_numpy(dtype=float)
    sig = cov.loc[exposures.index, exposures.index].to_numpy(dtype=float)
    sp = float(np.sqrt(max(w @ sig @ w, 0.0)))
    if sp == 0.0:
        raise ValueError("book has zero linear risk; reverse stress undefined")
    if (loss_target is None) == (radius is None):
        raise ValueError("specify exactly one of loss_target or radius")
    k = float(radius) if radius is not None else float(loss_target) / sp
    if k <= 0:
        raise ValueError("radius / loss_target must be positive")
    dx = -k * (sig @ w) / sp
    return pd.Series(dx, index=exposures.index), float(k * sp)


def reverse_stress_numerical(
    exposures: pd.Series, cov: pd.DataFrame, radius: float, seed: int = 0
) -> tuple[pd.Series, float]:
    """Numerical check of :func:`reverse_stress_linear`.

    Maximises the linear loss over the Mahalanobis ellipsoid by SLSQP from
    a seeded random start; used in tests to confirm the closed form.
    """
    w = exposures.to_numpy(dtype=float)
    sig = cov.loc[exposures.index, exposures.index].to_numpy(dtype=float)
    sig_inv = np.linalg.pinv(sig)

    def neg_loss(dx):
        return float(w @ dx)  # loss = -w'dx; minimise w'dx

    cons = {"type": "ineq", "fun": lambda dx: radius**2 - dx @ sig_inv @ dx}
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(w.size) * 1e-4
    res = minimize(neg_loss, x0, constraints=[cons], method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-14})
    dx = res.x
    return pd.Series(dx, index=exposures.index), float(-w @ dx)
