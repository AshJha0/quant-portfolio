"""Simulated power check for ``prereg/momentum-fx-g10.md``; synthetic data only.

Ground truth: nine currencies, daily log return = drift + noise (sd 0.6 %).
Each currency's drift is constant within a month and follows a monthly
AR(1) with persistence 0.9 and stationary annualised sd ``drift_sd``, so past
returns predict future ones (planted momentum). ``drift_sd = 0`` is the null.
Each rep simulates 63 lookback days plus 504 scored days (24 months), runs the
registered rule (``forward_studies.momentum_returns``, first-build cost on the
first scored day as in the evaluation) and applies the registered decision
rule, Sharpe - 1.645 x Lo SE > 0.

Run: ``python research/studies/momentum_power.py [--reps 200]``.
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
from qp_core.stats import lo_sharpe_se, sharpe_ratio  # noqa: E402

OUT = REPO / "research" / "power" / "momentum-fx-g10.json"
CCY = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"]
N_LOOK, N_TEST = 63, 504
DAILY_SD = 0.006
PHI = 0.9
DRIFT_SDS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16)


def simulate(rng: np.random.Generator, drift_sd: float) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=N_LOOK + N_TEST + 30)
    months = np.r_[0, np.cumsum(idx.month[1:] != idx.month[:-1])]
    m = np.zeros((months[-1] + 1, len(CCY)))
    m[0] = rng.normal(0, drift_sd, len(CCY))
    for k in range(1, len(m)):
        m[k] = PHI * m[k - 1] + np.sqrt(1 - PHI**2) * rng.normal(0, drift_sd, len(CCY))
    r = m[months] / 252 + rng.normal(0, DAILY_SD, (len(idx), len(CCY)))
    return pd.DataFrame(np.exp(np.cumsum(r, axis=0)), index=idx, columns=CCY)


def one_rep(rng: np.random.Generator, drift_sd: float) -> tuple[float, bool]:
    px = simulate(rng, drift_sd)
    r = fs.momentum_returns(px)
    w = fs.momentum_weights(px)
    start = len(px) - N_TEST
    scored = r.iloc[start - 1:].to_numpy().copy()
    scored[0] -= fs.MOMENTUM["cost_bp"] * 1e-4 * float(w.iloc[start - 1].abs().sum())
    sr = sharpe_ratio(scored)
    return sr, bool(sr - 1.645 * lo_sharpe_se(scored) > 0)


def run(reps: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cells = []
    for d in DRIFT_SDS:
        out = [one_rep(rng, d) for _ in range(reps)]
        cells.append({"drift_sd_annual": d, "truth": "null" if d == 0 else "planted momentum", "reps": reps,
                      "rejection_rate": float(np.mean([x[1] for x in out])),
                      "mean_sharpe": float(np.mean([x[0] for x in out]))})
    return {"what": "simulated power check for prereg/momentum-fx-g10.md; synthetic data only",
            "lookback_days": N_LOOK, "test_days": N_TEST, "daily_sd": DAILY_SD, "monthly_drift_phi": PHI,
            "cost_bp": fs.MOMENTUM["cost_bp"], "decision_rule": "Sharpe - 1.645 * Lo NW SE > 0",
            "seed": seed, "cells": cells}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args(argv)
    doc = run(a.reps)
    for c in doc["cells"]:
        print(json.dumps(c))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
