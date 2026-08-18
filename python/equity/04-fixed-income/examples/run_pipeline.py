"""End-to-end fixed income pipeline.

Bootstrap a curve from synthetic deposit/swap quotes, verify the 1e-10
round trip, price a govt+corp portfolio, produce the full risk report
(duration / convexity / DV01 / KRD ladder), run scenario P&L including
historical episodes, compare duration-approximation vs full revaluation,
and compute carry & roll-down.  Runs offline in a few seconds.
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fi_rates as fr
from fi_rates.data import market_quotes, sample_portfolio

pd.set_option("display.width", 120)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")

SETTLEMENT = dt.date(2026, 8, 18)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------ bootstrap
    section("1. BOOTSTRAP  (synthetic deposit + swap quotes, upward curve)")
    quotes = market_quotes("upward", seed=42)
    curve = fr.bootstrap_curve(quotes, interpolation="loglinear_df")
    rows = []
    for t in curve.times:
        rows.append(
            {
                "tenor_y": t,
                "df": float(np.asarray(curve.df(t))),
                "zero_%": 100 * float(np.asarray(curve.zero_rate(t))),
                "fwd_1y_%": 100 * curve.forward_rate(t, t + 1.0)
                if t + 1.0 <= curve.times[-1]
                else np.nan,
                "par_%": 100 * curve.par_rate(t) if abs(t - round(t)) < 1e-9 and t >= 1 else np.nan,
            }
        )
    print(pd.DataFrame(rows).set_index("tenor_y").round(6).to_string())

    section("2. ROUND TRIP  (reprice every input instrument off the curve)")
    errs = fr.reprice_instruments(quotes, curve)
    for ins, e in errs:
        kind = type(ins).__name__
        print(f"  {kind:8s} T={ins.pillar:5.2f}y  quote={ins.rate:+.6%}  "
              f"model-quote = {e:+.3e}")
    worst = max(abs(e) for _, e in errs)
    print(f"  max abs repricing error: {worst:.3e}  (tolerance 1e-10: "
          f"{'PASS' if worst < 1e-10 else 'FAIL'})")

    # -------------------------------------------------------------- pricing
    section("3. PORTFOLIO PRICING  (govt zero-spread + corp z-spread)")
    positions = sample_portfolio(SETTLEMENT, seed=42)
    tbl = []
    for pos in positions:
        dirty = fr.dirty_price_from_curve(pos.bond, SETTLEMENT, curve, pos.z_spread)
        acc = fr.accrued_interest(pos.bond, SETTLEMENT)
        clean = dirty - acc
        y = fr.ytm_from_price(pos.bond, SETTLEMENT, clean)
        zs = fr.z_spread_from_price(pos.bond, SETTLEMENT, curve, clean)
        tbl.append(
            {
                "position": pos.label,
                "qty": pos.quantity,
                "coupon_%": 100 * pos.bond.coupon,
                "clean": clean,
                "accrued": acc,
                "dirty": dirty,
                "ytm_%": 100 * y,
                "zspread_bp": 1e4 * zs,
                "mv": pos.quantity * dirty,
            }
        )
    print(pd.DataFrame(tbl).set_index("position").round(4).to_string())
    mv = fr.portfolio_value(positions, SETTLEMENT, curve)
    print(f"\n  total market value: {mv:,.2f}")

    # ----------------------------------------------------------------- risk
    section("4. RISK REPORT  (duration / convexity / DV01)")
    risk = fr.portfolio_risk(positions, SETTLEMENT, curve)
    print(risk.round(4).to_string())

    section("5. KEY-RATE DV01 LADDER  (2s/5s/10s/30s triangular bumps)")
    krd = fr.krd_report(positions, SETTLEMENT, curve)
    print(krd.round(4).to_string())
    parallel_dv01 = float(risk.loc["TOTAL", "dv01"])
    krd_sum = float(krd.loc["SUM", "key_rate_dv01"])
    print(f"\n  sum of KRDV01s: {krd_sum:,.4f}   parallel DV01: "
          f"{parallel_dv01:,.4f}   diff: {krd_sum - parallel_dv01:+.6f}  "
          f"({abs(krd_sum - parallel_dv01) / parallel_dv01:.2e} relative)")

    # ------------------------------------------------------------ scenarios
    section("6. SCENARIO P&L  (full revaluation vs duration+convexity estimate)")
    scen = [
        fr.parallel_scenario(+100),
        fr.parallel_scenario(-100),
        fr.steepener_scenario(-50, +50, name="steepener_50bp"),
        fr.steepener_scenario(+50, -50, name="flattener_50bp"),
        fr.butterfly_scenario(-25, +50, name="belly_cheapening_fly"),
        *fr.HISTORICAL_SCENARIOS.values(),
    ]
    pnl = fr.scenario_pnl_table(positions, SETTLEMENT, curve, scen)
    print(pnl.round(2).to_string())

    section("7. DURATION vs FULL REPRICING  (10y govt, Taylor error table)")
    govt10 = positions[2].bond
    clean10 = fr.clean_price_from_curve(govt10, SETTLEMENT, curve)
    y10 = fr.ytm_from_price(govt10, SETTLEMENT, clean10)
    err_tbl = fr.pnl_approximation_table(govt10, SETTLEMENT, y10)
    print(err_tbl.round(4).to_string())

    # ------------------------------------------------------ carry / rolldown
    section("8. CARRY & ROLL-DOWN  (1y horizon, static curve)")
    rows = []
    for pos in positions:
        horizon = SETTLEMENT + dt.timedelta(days=365)
        cr = fr.carry_rolldown(pos.bond, SETTLEMENT, horizon, curve, pos.z_spread)
        rows.append({"position": pos.label, **{k: v for k, v in cr.items()}})
    print(pd.DataFrame(rows).set_index("position").round(4).to_string())

    print(f"\npipeline completed in {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
