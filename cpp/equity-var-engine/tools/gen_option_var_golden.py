"""Golden values for option VaR parity across the four compiled VaR engines.

Run from the repo root:  python cpp/equity-var-engine/tools/gen_option_var_golden.py

The Python reference is eq_var.revaluation.full_revaluation_pnl /
fx_var.revaluation.full_revaluation_pnl followed by the S2 order-statistic
VaR / ES (eq_var._tail.var_es). The scenario panel is a closed-form
deterministic sequence, so every engine rebuilds it bit for bit without an
RNG; the tests in cpp/{equity,fx}-var-engine and rust/{equity,fx}-var-engine
paste the printed numbers.

Monte Carlo has no shared RNG stream, so the MC check is against a
closed-form target instead: a long call on one normal factor has P&L
monotone in the factor, hence VaR_alpha = -pnl(sigma z_alpha) exactly and
ES_alpha = -(1/alpha) int_0^alpha pnl(sigma Phi^{-1}(u)) du (computed here by
quadrature). The engines must land within 4 bootstrap standard errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python/equity/03-var-es-engine/src"))
sys.path.insert(0, str(ROOT / "python/fx/03-var-es-engine/src"))

from eq_var import revaluation as eqr  # noqa: E402
from eq_var._tail import var_es  # noqa: E402
from fx_var import revaluation as fxr  # noqa: E402

N = 250
ALPHA = 0.01
DT = 1.0 / 252.0


def panel() -> np.ndarray:
    i = np.arange(N, dtype=float)
    return np.column_stack(
        [
            0.02 * np.sin(1.3 * i + 0.4),
            0.015 * np.cos(0.7 * i) + 0.005 * np.sin(2.9 * i),
        ]
    )


def show(label: str, pnl: np.ndarray) -> None:
    v, e = var_es(pnl, ALPHA)
    print(f"{label}: var={v!r} es={e!r} pnl0={float(pnl[0])!r} pnl7={float(pnl[7])!r}")


def main() -> None:
    x = panel()
    # Equity: 10 long calls on factor 0, 5 short puts on factor 1, plus the
    # engine's linear convention exposures . r with w = (1000, -500).
    eq_legs = [
        eqr.OptionLeg(10.0, "call", 100.0, 105.0, 0.5, 0.03, 0.01, 0.25, 0),
        eqr.OptionLeg(-5.0, "put", 50.0, 48.0, 0.25, 0.03, 0.0, 0.30, 1),
    ]
    w = np.array([1000.0, -500.0])
    show("eq options only dt=0", eqr.full_revaluation_pnl(eq_legs, x))
    show("eq options + linear dt=0", eqr.full_revaluation_pnl(eq_legs, x) + x @ w)
    show("eq options + linear dt=1/252", eqr.full_revaluation_pnl(eq_legs, x, dt=DT) + x @ w)

    # FX (USD base): EURUSD call and short USDJPY-style put on factor 1, plus
    # a 1,000,000 EUR spot position (LinearLeg, P&L = A S (e^x - 1)).
    fx_legs = [
        fxr.OptionLeg(1.0e6, "call", 1.10, 1.12, 0.25, 0.045, 0.03, 0.08, 0),
        fxr.OptionLeg(-2.0e6, "put", 0.75, 0.74, 0.5, 0.045, 0.04, 0.10, 1),
    ]
    show("fx options only dt=0", fxr.full_revaluation_pnl(fx_legs, x))
    lin = [fxr.LinearLeg(1.0e6 * 1.10, 1.0, 0)]
    show("fx options + spot dt=1/252", fxr.full_revaluation_pnl(fx_legs + lin, x, dt=DT))

    # MC closed-form target: 10 long calls, one factor, daily sigma 1 %.
    sig = 0.01
    leg = eqr.OptionLeg(10.0, "call", 100.0, 100.0, 0.25, 0.02, 0.0, 0.2, 0)

    def pnl_at(z: float) -> float:
        return float(eqr.full_revaluation_pnl([leg], np.array([[sig * z]]))[0])

    var_t = -pnl_at(norm.ppf(ALPHA))
    es_t = -integrate.quad(lambda u: pnl_at(norm.ppf(u)), 0.0, ALPHA, limit=200)[0] / ALPHA
    print(f"mc target eq: var={var_t!r} es={es_t!r}")
    fleg = fxr.OptionLeg(1.0e6, "call", 1.10, 1.10, 0.25, 0.045, 0.03, 0.08, 0)

    def fpnl_at(z: float) -> float:
        return float(fxr.full_revaluation_pnl([fleg], np.array([[0.006 * z]]))[0])

    var_f = -fpnl_at(norm.ppf(ALPHA))
    es_f = -integrate.quad(lambda u: fpnl_at(norm.ppf(u)), 0.0, ALPHA, limit=200)[0] / ALPHA
    print(f"mc target fx (sigma 0.6 %): var={var_f!r} es={es_f!r}")


if __name__ == "__main__":
    main()
