"""Write the cross-language parity fixture for the capital and xVA kernels.

``python -m qp_core.xva._parity OUT`` writes one record per line: a kernel
name, its inputs and the Python result, every float as ``repr`` (shortest
round-trip). The Rust test ``rust/core/tests/xva_capital_parity.rs`` and the
C++ test ``cpp/core/tests/test_xva_capital.cpp`` read the same file and must
reproduce each result to 1e-13 relative (1e-15 absolute near zero).
``tests/test_xva_parity_fixture.py`` fails if the committed file drifts from
what this module generates.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from ..capital import frtb_sbm as sbm
from ..capital import saccr as s
from ..market import DiscountCurve
from .adjustments import HazardCurve, cva, fva
from .exposure import eepe, pfe_index
from .hull_white import HullWhite


def _f(x: float) -> str:
    return repr(float(x))


def records() -> list[str]:
    out: list[str] = []
    rng = np.random.default_rng(20261103)
    for s0, e0 in [(0.0, 10.0), (0.0, 4.0), (1.0, 11.0), (0.0, 0.01), (2.5, 30.0)]:
        out.append(f"SD {_f(s0)} {_f(e0)} {_f(s.supervisory_duration(s0, e0))}")
    for p, k, v, t, call, bought, sh in [
        (0.06, 0.05, 0.5, 1.0, 0, 1, 0.0),
        (0.06, 0.05, 0.5, 1.0, 1, 0, 0.0),
        (-0.002, 0.001, 0.5, 2.0, 1, 1, 0.01),
        (100.0, 120.0, 0.32, 0.25, 0, 0, 0.0),
    ]:
        dl = s.supervisory_delta_option(p, k, v, t, call=bool(call), bought=bool(bought), shift=sh)
        out.append(f"DELTA {_f(p)} {_f(k)} {_f(v)} {_f(t)} {call} {bought} {_f(sh)} {_f(dl)}")
    for m, mp in [(0.75, -1.0), (5.0, -1.0), (0.001, -1.0), (5.0, 14.0), (0.5, 10.0)]:
        out.append(f"MF {_f(m)} {_f(mp)} {_f(s.maturity_factor(m, mpor=None if mp < 0 else mp))}")
    for mv, mc, ma in [(-20.0, 0.0, 282.2), (80.0, 200.0, 1401.0), (10.0, 0.0, 5.0), (0.0, 0.0, 0.0)]:
        out.append(f"MULT {_f(mv)} {_f(mc)} {_f(ma)} {_f(s.multiplier(mv, mc, ma))}")
    for dd in [(0.0, -36254.0, 78694.0), (1.0, 2.0, 3.0), (-5.0, 7.0, -11.0)]:
        out.append(f"IREN {' '.join(_f(x) for x in dd)} {_f(s.ir_effective_notional(*dd))}")
    for _ in range(4):
        n = int(rng.integers(1, 6))
        aa = [float(x) for x in rng.normal(0, 100, n)]
        rr = [float(x) for x in rng.choice([0.4, 0.5, 0.8], n)]
        res = s.single_factor_addon(aa, rr)
        out.append(f"SFA {n} {' '.join(_f(x) for x in aa)} {' '.join(_f(x) for x in rr)} {_f(res)}")
    curve = DiscountCurve(
        (0.5, 1.0, 5.0, 30.0),
        tuple(math.exp(-r * t) for r, t in [(0.02, 0.5), (0.025, 1.0), (0.03, 5.0), (0.035, 30.0)]),
    )
    for ha, sig in [(0.05, 0.01), (0.2, 0.015), (0.01, 0.005)]:
        hw = HullWhite(curve, ha, sig)
        for T, S, X, call in [(1.0, 3.0, 0.95, 1), (2.0, 10.0, 0.8, 0), (0.5, 0.75, 0.99, 1)]:
            zv = hw.zcb_option(T, S, X, call=bool(call))
            out.append(
                f"ZBO {_f(ha)} {_f(sig)} {_f(T)} {_f(S)} {_f(hw.p0(T))} {_f(hw.p0(S))} {_f(X)} {call} {_f(zv)}"
            )
        for t0, n, k, payer in [(1.0, 5, 0.03, 1), (2.0, 8, 0.025, 0), (5.0, 10, 0.04, 1)]:
            pay = [t0 + 0.5 * (i + 1) for i in range(n)]
            acc = [0.5] * n
            sv = hw.swaption(t0, pay, acc, k, payer=bool(payer))
            out.append(
                f"SWPT {_f(ha)} {_f(sig)} {_f(t0)} {_f(hw.p0(t0))} {n} "
                + " ".join(_f(x) for x in pay)
                + " "
                + " ".join(_f(hw.p0(x)) for x in pay)
                + " "
                + " ".join(_f(x) for x in acc)
                + f" {_f(k)} {payer} {_f(sv)}"
            )
    for _ in range(3):
        m = int(rng.integers(3, 9))
        tg: list[float] = [float(x) for x in np.concatenate([[0.0], np.cumsum(rng.uniform(0.1, 1.0, m - 1))])]
        ee = [float(x) for x in rng.uniform(0, 5, m)]
        ene = [-float(x) for x in rng.uniform(0, 5, m)]
        hc = HazardCurve((1.0, 3.0), (float(rng.uniform(0, 0.05)), float(rng.uniform(0, 0.05))))
        hb = HazardCurve((2.0,), (float(rng.uniform(0, 0.03)),))
        rec = 0.4
        c1 = cva(tg, ee, hc, rec)
        c2 = cva(tg, ee, hc, rec, bank=hb)
        fca, fba = fva(tg, ee, ene, 0.01, 0.004, hc, hb)
        hz = f"{_f(hc.times[0])} {_f(hc.times[1])} {_f(hc.hazards[0])} {_f(hc.hazards[1])} {_f(hb.times[0])} {_f(hb.hazards[0])}"
        arr = (
            " ".join(_f(x) for x in tg)
            + " "
            + " ".join(_f(x) for x in ee)
            + " "
            + " ".join(_f(x) for x in ene)
        )
        out.append(f"XVA {m} {arr} {hz} {_f(c1)} {_f(c2)} {_f(fca)} {_f(fba)}")
        mat = float(rng.uniform(0.5, 3.0))
        out.append(
            f"EEPE {m} "
            + " ".join(_f(x) for x in tg)
            + " "
            + " ".join(_f(x) for x in ee)
            + f" {_f(mat)} {_f(eepe(tg, ee, mat))}"
        )
    for n, q in [(1000, 0.95), (100, 0.99), (7, 0.5), (1, 0.975)]:
        out.append(f"PFE {n} {_f(q)} {pfe_index(n, q)}")
    for tk, tl, same in [(0.25, 30.0, 1), (1.0, 5.0, 1), (1.0, 5.0, 0), (2.0, 3.0, 1), (10.0, 10.0, 0)]:
        out.append(f"GRHO {_f(tk)} {_f(tl)} {same} {_f(sbm.girr_delta_rho(tk, tl, same_curve=bool(same)))}")
    for rho in (0.0, 0.3, 0.5, 0.6, 0.97, 1.0):
        for i, sc in enumerate(sbm.SCENARIOS):
            out.append(f"SCEN {_f(rho)} {i} {_f(sbm.scenario_corr(rho, sc))}")
    for _ in range(4):
        n = int(rng.integers(1, 6))
        ws = [float(x) for x in rng.normal(0, 50, n)]
        tn = sorted(float(x) for x in rng.choice(sbm.GIRR_TENORS, n, replace=False))
        rm = np.array([[sbm.girr_delta_rho(a, b) for b in tn] for a in tn])
        out.append(
            f"KB {n} "
            + " ".join(_f(x) for x in ws)
            + " "
            + " ".join(_f(x) for x in rm.ravel())
            + f" {_f(sbm.within_bucket(ws, rm))}"
        )
    for kb, sb, g in [
        ([7.0, 7.0], [10.0, -10.0], 0.5),
        ([7.0, 7.0], [5.0, -5.0], 0.5),
        ([3.0, 4.0, 5.0], [2.0, 4.0, -5.0], 0.6),
    ]:
        gm = np.full((len(kb), len(kb)), g)
        out.append(
            f"ACROSS {len(kb)} "
            + " ".join(_f(x) for x in kb)
            + " "
            + " ".join(_f(x) for x in sb)
            + f" {_f(g)} {_f(sbm.across_buckets(kb, sb, gm))}"
        )
    return out


def main(argv: list[str]) -> int:
    text = "# generated by python -m qp_core.xva._parity; do not edit\n" + "\n".join(records()) + "\n"
    with open(argv[1], "w", encoding="ascii", newline="\n") as fh:
        fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
