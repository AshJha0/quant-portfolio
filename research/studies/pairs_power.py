"""Simulated power check for the pairs v2 registration (no real data).

Ground truth: pairs that ARE cointegrated, log y = c + b log x + s with s a
stationary AR(1), and log x a random walk at a EUR-like level (log price
near 0) or a JPY-like level (USD per JPY near 0.0067, log price near -5).
Each replication builds a 3-year formation window and a 2-year test window
(the registration's 24-month evaluation), runs the registered selection
(Engle-Granger AND Johansen) on the formation window and trades the test
window, for both signals:

- ``level``: the registered level z (``forward_studies.pair_returns_level``);
- ``kalman_v1``: the signed design's Kalman one-step z (``fx_studies.pair_returns``).

The portfolio is equal weight over the selected pairs among ``n_pairs``
independent ground-truth pairs, as the study does. Reported per cell: the
selection rate, the share of test days with |z| > 2, entries per pair-year,
mean annualised net Sharpe and the rejection rate of the registered decision
rule (Sharpe - 1.645 x Lo SE > 0). A null cell (s a random walk, no
cointegration) gives the size.

The point is the one a reviewer raised: on cointegrated pairs the Kalman
z of the signed design almost never reaches 2 (its intercept follows the
spread), so that test has almost no power whatever the truth. The
registered design has to show it can see a true edge before it is signed.

Run: ``python research/studies/pairs_power.py [--reps N]``; writes
``research/power/pairs_v2_power.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "research" / "tools"), str(REPO / "data" / "fetch")]

import forward_studies as fs  # noqa: E402
import fx_studies  # noqa: E402
from fx_pairs.cointegration import engle_granger  # noqa: E402
from models import johansen_trace  # noqa: E402
from qp_core.stats import lo_sharpe_se, sharpe_ratio  # noqa: E402

OUT = REPO / "research" / "power" / "pairs_v2_power.json"
N_FORM, N_TEST = 756, 504
LEVELS = {"eur_like": 0.1, "jpy_like": float(np.log(0.0067))}


def simulate_pair(rng: np.random.Generator, half_life: float | None, level: float, spread_sd: float = 0.01,
                  beta: float = 0.8, x_sd: float = 0.006) -> tuple[np.ndarray, np.ndarray]:
    """One ground-truth pair of log prices. ``half_life=None`` makes the
    spread a random walk with the same daily innovation as the AR(1) at a
    10-day half-life (the null: not cointegrated)."""
    n = N_FORM + N_TEST
    lx = level + np.cumsum(rng.normal(0.0, x_sd, n))
    if half_life is None:
        phi, eps_sd = 1.0, spread_sd * np.sqrt(1 - 0.5 ** (2 / 10))
    else:
        phi = 0.5 ** (1.0 / half_life)
        eps_sd = spread_sd * np.sqrt(1 - phi**2)
    s = np.zeros(n)
    s[0] = rng.normal(0.0, spread_sd)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + rng.normal(0.0, eps_sd)
    ly = 0.05 - beta * level + beta * lx + s  # y sits near a EUR-like level whatever x's level
    return ly, lx


def selected(ly: np.ndarray, lx: np.ndarray) -> bool:
    fy, fx = ly[:N_FORM], lx[:N_FORM]
    return bool(engle_granger(fy, fx).cointegrated and johansen_trace(np.column_stack([fy, fx]), 1).rank >= 1)


def one_rep(rng: np.random.Generator, half_life: float | None, level: float, n_pairs: int) -> dict:
    test = np.r_[np.zeros(N_FORM, bool), np.ones(N_TEST, bool)]
    rets: dict[str, list[np.ndarray]] = {"level": [], "kalman_v1": []}
    zfrac: dict[str, list[float]] = {"level": [], "kalman_v1": []}
    entries = {"level": 0, "kalman_v1": 0}
    n_sel = 0
    for _ in range(n_pairs):
        ly, lx = simulate_pair(rng, half_life, level)
        if not selected(ly, lx):
            continue
        n_sel += 1
        r, act = fs.pair_returns_level(ly, lx, test)
        rets["level"].append(r)
        zfrac["level"].append(act["entry_signal_days"] / N_TEST)
        entries["level"] += act["entries"]
        kact: dict = {}
        rets["kalman_v1"].append(fx_studies.pair_returns(pd.Series(ly), pd.Series(lx), test, fx_studies.PAIRS, kact))
        zfrac["kalman_v1"].append(kact["entry_signal_days"] / N_TEST)
        entries["kalman_v1"] += kact["entries"]
    out: dict = {"n_selected": n_sel}
    for k, rr in rets.items():
        port = np.mean(rr, axis=0) if rr else np.zeros(N_TEST)
        if np.count_nonzero(port) < 3 or not np.std(port) > 0:
            sr, reject = 0.0, False
        else:
            sr = sharpe_ratio(port)
            reject = bool(sr - 1.645 * lo_sharpe_se(port) > 0)
        out[k] = {"sharpe": sr, "reject": reject, "zfrac": float(np.mean(zfrac[k])) if zfrac[k] else 0.0,
                  "entries": entries[k]}
    return out


def run_cell(half_life: float | None, level: float, reps: int, n_pairs: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    reps_out = [one_rep(rng, half_life, level, n_pairs) for _ in range(reps)]
    sel = sum(r["n_selected"] for r in reps_out)
    cell: dict = {"half_life_days": half_life, "log_level_x": level, "reps": reps, "n_pairs": n_pairs,
                  "selection_rate": sel / (reps * n_pairs)}
    for k in ("level", "kalman_v1"):
        cell[k] = {"rejection_rate": float(np.mean([r[k]["reject"] for r in reps_out])),
                   "mean_sharpe": float(np.mean([r[k]["sharpe"] for r in reps_out])),
                   "share_test_days_abs_z_above_2": float(np.mean([r[k]["zfrac"] for r in reps_out
                                                                   if r["n_selected"]] or [0.0])),
                   "entries_per_selected_pair_year": (sum(r[k]["entries"] for r in reps_out) / sel / 2.0
                                                      if sel else 0.0)}
    return cell


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args(argv)
    cells = []
    for i, (hl, (lname, lvl)) in enumerate([(10, ("eur_like", LEVELS["eur_like"])),
                                           (10, ("jpy_like", LEVELS["jpy_like"])),
                                           (30, ("eur_like", LEVELS["eur_like"])),
                                           (30, ("jpy_like", LEVELS["jpy_like"])),
                                           (None, ("eur_like", LEVELS["eur_like"])),
                                           (None, ("jpy_like", LEVELS["jpy_like"]))]):
        c = run_cell(hl, lvl, a.reps, seed=i)
        c["level_name"] = lname
        cells.append(c)
        print(json.dumps(c))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {"what": "simulated power check for prereg/pairs-fx-g10-v2.md; synthetic data only",
           "formation_days": N_FORM, "test_days": N_TEST, "spread_sd": 0.01, "beta": 0.8, "x_daily_sd": 0.006,
           "cost_bp_per_leg": fs.PAIRS_V2["cost_bp"], "decision_rule": "Sharpe - 1.645 * Lo NW SE > 0",
           "cells": cells}
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
