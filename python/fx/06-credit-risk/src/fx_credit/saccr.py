"""SA-CCR exposure at default for FX derivative netting sets.

Implements the standardised approach for counterparty credit risk for the
foreign-exchange asset class, Basel Framework CRE52 (in force 1 Jan 2023;
originally BCBS 279, March 2014, revised April 2014):

    EAD     = alpha * (RC + PFE),                 alpha = 1.4      (CRE52.1)
    RC      = max(V - C, 0)                       unmargined       (CRE52.10)
    RC      = max(V - C, TH + MTA - NICA, 0)      margined         (CRE52.18)
    PFE     = multiplier * AddOn                                   (CRE52.20)
    mult    = min(1, F + (1 - F) exp((V - C) / (2 (1 - F) AddOn))), F = 5%
                                                                   (CRE52.23)
    AddOn   = sum over currency pairs of SF * |sum_i delta_i d_i MF_i|,
              SF = 4%                                  (CRE52.58-CRE52.59)
    MF      = sqrt(min(max(M, 10 days), 1 year) / 1 year)  unmargined
    MF      = 1.5 sqrt(MPoR / 1 year)                      margined
                                                       (CRE52.48, CRE52.52)

d_i is the adjusted notional: the foreign-currency leg converted to the
domestic currency (CRE52.37), and delta_i = +1 for a long and -1 for a
short position in the first currency of the pair (CRE52.38). A year is
250 business days for MPoR.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

ALPHA = 1.4
SF_FX = 0.04
MULT_FLOOR = 0.05


@dataclass(frozen=True)
class FxTrade:
    pair: str
    notional_domestic: float
    long: bool
    maturity_years: float

    def __post_init__(self) -> None:
        if self.notional_domestic <= 0.0 or self.maturity_years < 0.0:
            raise ValueError("notional must be positive and maturity non-negative")


def maturity_factor(maturity_years: float, margined: bool = False, mpor_days: int = 10) -> float:
    if margined:
        if mpor_days < 1:
            raise ValueError("mpor_days must be at least 1")
        return 1.5 * float(np.sqrt(mpor_days / 250.0))
    m = min(max(maturity_years, 10.0 / 250.0), 1.0)
    return float(np.sqrt(m))


def fx_addon(trades: list[FxTrade], margined: bool = False, mpor_days: int = 10) -> float:
    """Aggregate FX add-on: one hedging set per currency pair, full offset inside it."""
    eff: dict[str, float] = defaultdict(float)
    for tr in trades:
        delta = 1.0 if tr.long else -1.0
        eff[tr.pair] += delta * tr.notional_domestic * maturity_factor(tr.maturity_years, margined, mpor_days)
    return SF_FX * float(sum(abs(x) for x in eff.values()))


def pfe_multiplier(v_minus_c: float, addon: float) -> float:
    if addon <= 0.0:
        return 1.0
    return min(
        1.0, MULT_FLOOR + (1.0 - MULT_FLOOR) * float(np.exp(v_minus_c / (2.0 * (1.0 - MULT_FLOOR) * addon)))
    )


@dataclass(frozen=True)
class SaccrResult:
    rc: float
    addon: float
    multiplier: float
    pfe: float
    ead: float


def saccr_ead(
    trades: list[FxTrade],
    mtm: float,
    collateral: float = 0.0,
    margined: bool = False,
    threshold: float = 0.0,
    mta: float = 0.0,
    nica: float = 0.0,
    mpor_days: int = 10,
) -> SaccrResult:
    """SA-CCR EAD of one FX netting set (module docstring for each formula).

    For a margined set the EAD is capped at the unmargined EAD (CRE52.2).
    """
    vc = mtm - collateral
    addon = fx_addon(trades, margined, mpor_days)
    mult = pfe_multiplier(vc, addon)
    rc = max(vc, threshold + mta - nica, 0.0) if margined else max(vc, 0.0)
    pfe = mult * addon
    ead = ALPHA * (rc + pfe)
    if margined:
        unm = saccr_ead(trades, mtm, collateral, margined=False)
        ead = min(ead, unm.ead)
    return SaccrResult(rc, addon, mult, pfe, ead)
