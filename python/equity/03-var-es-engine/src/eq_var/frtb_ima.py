"""FRTB internal models approach: liquidity-horizon ES, stressed ES, IMCC,
capital, desk backtests and the P&L attribution (PLA) test.

Plain idea
----------
FRTB replaces 10-day 99 % VaR with 97.5 % ES, and asks how long each risk
factor takes to exit in a stress. Factors are bucketed into liquidity
horizons of 10, 20, 40, 60 or 120 days. The bank computes 10-day ES with
progressively fewer factors shocked (all factors, then only those with a
horizon of 20 days or more, and so on) and stacks the pieces with square-root
time scaling. The resulting ES is calibrated to a stress period, combined
across risk classes, and backed by two desk-level tests: VaR backtesting and
a check that the risk model's P&L tracks the front office's P&L.

Formal statement (BCBS, *Minimum capital requirements for market risk*,
January 2019, rev. February 2019; consolidated framework chapters MAR32 and
MAR33)
----------------------------------------------------------------------
* Liquidity-horizon cascade (MAR33.4), base horizon T = 10 days,
  LH = (10, 20, 40, 60, 120):

      ES = sqrt( ES_T(P)^2 + sum_{j>=2} ( ES_T(P, j) sqrt((LH_j - LH_{j-1}) / T) )^2 )

  where ``ES_T(P, j)`` shocks only factors with horizon ``>= LH_j``.
* Stressed calibration (MAR33.5): ``ES = ES_RS * max(ES_FC / ES_RC, 1)`` with
  ``RS`` the reduced factor set in the stress period, ``FC`` the full set and
  ``RC`` the reduced set in the current period.
* IMCC (MAR33.15): ``rho ES_unconstrained + (1 - rho) sum_k ES_k`` with
  ``rho = 0.5``, the sum over the risk-class ES values computed alone.
* Capital (MAR33.41): ``max(IMCC_{t-1} + SES_{t-1},
  m_c avg60(IMCC) + avg60(SES))``, ``m_c = 1.5 + plus factor``. The plus
  factor comes from the count of 99 % VaR exceptions over 250 days (MAR99,
  Table 2): 0-4 -> 0, 5 -> 0.20, 6 -> 0.26, 7 -> 0.33, 8 -> 0.38, 9 -> 0.42,
  10+ -> 0.50.
* Desk backtest (MAR32.5): a desk stays on IMA if, over the last 250 days,
  it has at most 12 exceptions at 99 % VaR and at most 30 at 97.5 % VaR.
* PLA test (MAR32.42-32.44): Spearman rank correlation and the two-sample
  Kolmogorov-Smirnov statistic between hypothetical P&L (HPL) and
  risk-theoretical P&L (RTPL). Green if Spearman >= 0.80 and KS <= 0.09;
  red if Spearman < 0.70 or KS > 0.12; amber otherwise.

The thresholds and formulas above are checked against our reading of the
standard; the paragraph numbers have not yet been re-checked against the
current bis.org consolidated text, so treat them as pointers.

What this module does not do: the non-modellable risk factor charge (SES)
needs the risk-factor eligibility test and stress scenarios per factor. The
caller passes SES in; we only add it. The default risk charge is out of scope.

Worked examples: ``tests/test_frtb_ima.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qp_core.model import model

from . import _r3checks as chk

LIQUIDITY_HORIZONS: tuple[int, ...] = (10, 20, 40, 60, 120)
BASE_HORIZON: int = 10
RHO_IMCC: float = 0.5
PLA_SPEARMAN_GREEN, PLA_SPEARMAN_RED = 0.80, 0.70
PLA_KS_GREEN, PLA_KS_RED = 0.09, 0.12
DESK_MAX_EXCEPTIONS_99, DESK_MAX_EXCEPTIONS_975 = 12, 30
#: MAR99 Table 2: plus factor by number of 99 % exceptions in 250 days.
PLUS_FACTOR: dict[int, float] = {5: 0.20, 6: 0.26, 7: 0.33, 8: 0.38, 9: 0.42}


def liquidity_adjusted_es(es_by_horizon) -> float:
    """MAR33.4 cascade.

    ``es_by_horizon[j]`` is the 10-day ES with only the factors whose
    liquidity horizon is ``>= LIQUIDITY_HORIZONS[j]`` shocked; entry 0 is the
    full-book ES. Pass fewer than five entries when the top buckets are empty.
    """
    es = chk.array("es_by_horizon", es_by_horizon)
    if es.size > len(LIQUIDITY_HORIZONS):
        raise ValueError(f"at most {len(LIQUIDITY_HORIZONS)} horizons, got {es.size}")
    if (es < 0).any():
        raise ValueError("ES values must be >= 0")
    lh = np.asarray(LIQUIDITY_HORIZONS[: es.size], dtype=float)
    scale = np.ones(es.size)
    scale[1:] = np.sqrt(np.diff(lh) / BASE_HORIZON)
    return float(np.sqrt(np.sum((es * scale) ** 2)))


def stressed_es(es_reduced_stress: float, es_full_current: float,
                es_reduced_current: float) -> float:
    """MAR33.5: scale the reduced-set stress ES by ``max(FC / RC, 1)``."""
    rs = chk.scalar("es_reduced_stress", es_reduced_stress, lo=0.0)
    fc = chk.scalar("es_full_current", es_full_current, lo=0.0)
    rc = chk.scalar("es_reduced_current", es_reduced_current, lo=0.0, strict=True)
    return rs * max(fc / rc, 1.0)


def imcc(es_unconstrained: float, es_by_risk_class, rho: float = RHO_IMCC) -> float:
    """MAR33.15: ``rho ES_all + (1 - rho) sum(ES_class)``."""
    u = chk.scalar("es_unconstrained", es_unconstrained, lo=0.0)
    c = chk.array("es_by_risk_class", es_by_risk_class)
    r = chk.scalar("rho", rho, lo=0.0)
    if r > 1.0 or (c < 0).any():
        raise ValueError("rho must be in [0, 1] and class ES values >= 0")
    return r * u + (1.0 - r) * float(c.sum())


def plus_factor(exceptions_99: int) -> float:
    """MAR99 Table 2 plus factor for a count of 99 % VaR exceptions."""
    n = chk.count("exceptions_99", exceptions_99, lo=0)
    if n >= 10:
        return 0.50
    return PLUS_FACTOR.get(n, 0.0)


@model(id="EQ-VAR-FRTB-IMA")
def ima_capital(imcc_history, ses_history, exceptions_99: int = 0) -> float:
    """MAR33.41 capital from the last 60 daily IMCC and SES values (latest last)."""
    i = chk.array("imcc_history", imcc_history, min_size=60)[-60:]
    s = chk.array("ses_history", ses_history, min_size=60)[-60:]
    mc = 1.5 + plus_factor(exceptions_99)
    return float(max(i[-1] + s[-1], mc * i.mean() + s.mean()))


@dataclass(frozen=True)
class DeskBacktest:
    exceptions_99: int
    exceptions_975: int
    eligible: bool


def desk_backtest(pnl, var_99, var_975) -> DeskBacktest:
    """MAR32.5 desk test on the last 250 days.

    ``var_*`` are positive one-day VaR forecasts aligned with ``pnl``; an
    exception is ``pnl < -VaR``. Run it once on actual and once on
    hypothetical P&L; the desk fails if either run fails.
    """
    p = chk.array("pnl", pnl, min_size=250)[-250:]
    v99 = chk.array("var_99", var_99, min_size=250)[-250:]
    v975 = chk.array("var_975", var_975, min_size=250)[-250:]
    n99 = int((p < -v99).sum())
    n975 = int((p < -v975).sum())
    return DeskBacktest(n99, n975, n99 <= DESK_MAX_EXCEPTIONS_99 and n975 <= DESK_MAX_EXCEPTIONS_975)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(x.size)
    r[order] = np.arange(1, x.size + 1, dtype=float)
    # average ranks over ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / cnt[inv]


def spearman(x, y) -> float:
    """Spearman rank correlation with average ranks for ties."""
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx @ rx) * (ry @ ry))
    if den == 0.0:
        raise ValueError("Spearman correlation undefined for constant input")
    return float(rx @ ry / den)


def ks_statistic(x, y) -> float:
    """Two-sample Kolmogorov-Smirnov statistic ``sup |F_x - F_y|``."""
    x, y = np.sort(np.asarray(x, float)), np.sort(np.asarray(y, float))
    grid = np.concatenate([x, y])
    fx = np.searchsorted(x, grid, side="right") / x.size
    fy = np.searchsorted(y, grid, side="right") / y.size
    return float(np.abs(fx - fy).max())


@dataclass(frozen=True)
class PLAResult:
    spearman: float
    ks: float
    zone: str  # "green" | "amber" | "red"


def pla_test(hpl, rtpl) -> PLAResult:
    """MAR32.42-32.44 PLA test on aligned daily HPL and RTPL (>= 250 days used)."""
    h = chk.array("hpl", hpl, min_size=250)[-250:]
    r = chk.array("rtpl", rtpl, min_size=250)[-250:]
    rho, ks = spearman(h, r), ks_statistic(h, r)
    if rho < PLA_SPEARMAN_RED or ks > PLA_KS_RED:
        zone = "red"
    elif rho >= PLA_SPEARMAN_GREEN and ks <= PLA_KS_GREEN:
        zone = "green"
    else:
        zone = "amber"
    return PLAResult(rho, ks, zone)
