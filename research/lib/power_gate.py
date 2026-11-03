"""Power gate for confirmatory tests, redesigned (W6).

Plain idea: a test that would miss a real effect most of the time is not
worth registering as confirmatory. The old gate asked for 80% power on one
currency basket over two years, which no realistic Sharpe ratio can reach,
so every study failed it. The new gate does two things:

1. A single-series forward test is ``monitoring`` only. It is logged and
   scored, but it never confirms or refutes anything.
2. A confirmatory test must be a pooled cross-sectional panel (30 or more
   currencies, a Fama-MacBeth slope t-statistic, 1999-2015 in-sample and
   2016-2026 hold-out). It passes when power at the post-2010 literature
   Sharpe, for that design, is at least 80%.

Formal statement. For a one-sided test at level ``alpha`` on a mean with
annualised Sharpe ``SR`` over ``Y`` years, the t-statistic is approximately
``N(SR sqrt(Y), 1)``, so power ``= Phi(SR sqrt(Y) - z_{1-alpha})``. For a
Fama-MacBeth slope the per-period slope series has Sharpe
``IC sqrt(N_eff)`` per month (Grinold and Kahn 2000, the fundamental law,
with ``N_eff`` the cross-section after correlation), so the annual Sharpe is
``IC sqrt(N_eff) sqrt(12)`` (Fama and MacBeth 1973).

Worked examples (``test_power_gate.py``):

* the momentum basket, two years, annual Sharpe 0.3 (post-2010 literature
  range; Menkhoff et al. 2012 JFE report weaker momentum after the 1990s):
  power ``Phi(0.3 sqrt(2) - 1.645) = Phi(-1.221) = 0.111``, blocked. The
  simulated 13% in ``research/power/momentum-fx-g10.json`` is the same story;
* a 30-currency panel, monthly IC 0.04, ``N_eff = 15``, 10.9 years:
  annual SR ``0.04 sqrt(15) sqrt(12) = 0.537``; power
  ``Phi(0.537 sqrt(10.9) - 1.645) = Phi(0.128) = 0.551``, blocked; with IC 0.06
  the SR is 0.805 and power ``Phi(1.013) = 0.844``, passes.

The IC and N_eff come from the registration and must cite a source.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from scipy import stats as st

GATE = 0.80
MIN_PANEL = 30


@dataclass(frozen=True)
class GateResult:
    verdict: str  # "confirmatory", "blocked" or "monitoring"
    power: float | None
    reason: str


def analytic_power(sr_annual: float, years: float, alpha: float = 0.05) -> float:
    """One-sided power of a mean test with annual Sharpe ``sr_annual`` over ``years``."""
    if years <= 0:
        raise ValueError("years must be positive")
    return float(st.norm.cdf(sr_annual * math.sqrt(years) - st.norm.ppf(1 - alpha)))


def panel_slope_sharpe(ic_monthly: float, n_eff: float) -> float:
    """Annual Sharpe of a Fama-MacBeth slope series under the fundamental law."""
    if n_eff <= 0:
        raise ValueError("n_eff must be positive")
    return float(ic_monthly * math.sqrt(n_eff) * math.sqrt(12.0))


def gate(design: str, *, n_currencies: int = 0, ic_monthly: float | None = None,
         n_eff: float | None = None, holdout_years: float | None = None,
         alpha: float = 0.05) -> GateResult:
    """Apply the gate to a registered design."""
    if design == "single_series":
        return GateResult("monitoring", None,
                          "single-series forward tests are monitoring only (W6)")
    if design != "panel_fama_macbeth":
        raise ValueError(f"unknown design {design!r}")
    if n_currencies < MIN_PANEL:
        return GateResult("blocked", None, f"panel has {n_currencies} < {MIN_PANEL} currencies")
    if ic_monthly is None or n_eff is None or holdout_years is None:
        raise ValueError("a panel design needs ic_monthly, n_eff and holdout_years")
    p = analytic_power(panel_slope_sharpe(ic_monthly, n_eff), holdout_years, alpha)
    if p >= GATE:
        return GateResult("confirmatory", p, f"power {p:.3f} >= {GATE}")
    return GateResult("blocked", p, f"power {p:.3f} < {GATE}")


def gate_from_simulation(path: Path, literature_cell: float) -> GateResult:
    """Read a simulated power file (``research/power/*.json``) and gate it at
    the cell whose planted ``drift_sd_annual`` equals ``literature_cell``.
    Such files are single-series designs, so the best verdict is monitoring."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = [c for c in doc["cells"] if c.get("drift_sd_annual") == literature_cell]
    if not cells:
        raise ValueError(f"no cell at drift_sd_annual={literature_cell}")
    p = float(cells[0]["rejection_rate"])
    if p >= GATE:
        return GateResult("monitoring", p, "single series; power is adequate but the design "
                          "is monitoring only")
    return GateResult("blocked", p, f"power {p:.3f} < {GATE}; single series, monitoring only")
