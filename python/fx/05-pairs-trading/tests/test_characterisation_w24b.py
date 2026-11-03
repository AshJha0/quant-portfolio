"""W24b characterisation corpus for the move onto the qp_bt core.

``golden/characterisation_w24b_before.json`` is the output captured before
the migration (Round 12, lane r12-l4-btfills). ``characterisation_w24b.json``
pins the migrated output to 1e-12. The two differ by the wrong-answer fixes
listed in the module docstring of ``backtest.py``; ``BEFORE_AFTER`` pins the
summed series before and after so the documented numbers cannot drift, and
each fix has a hand fixture in the package tests. Regenerate only on
purpose: QP_REGEN_GOLDEN=1.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fx_pairs.backtest import ExecutionConfig, run_backtest, walk_forward_backtest
from qp_core.stats import Financing

GOLDEN = Path(__file__).parent / "golden" / "characterisation_w24b.json"
BEFORE = Path(__file__).parent / "golden" / "characterisation_w24b_before.json"
# key: (sum before migration, sum after), filled from the corpus
BEFORE_AFTER: dict[str, tuple[float, float]] = {
    "auction/carry": (-7.863722708182306e-05, -7.27138169237458e-05),
    "auction/cost": (-0.03481492802796217, -0.034825286497679825),
    "auction/total": (0.017276753381795833, 0.01774168401331547),
    "next_fin/carry": (-80.34110808511139, -82.76228021598169),
    "next_fin/cost": (-12501.241335511211, -12502.732727521012),
    "next_fin/total": (9318.68838749192, 9728.79851813406),
    "next_nofin_linear/carry": (-7.951337817616159e-05, -7.778639153279101e-05),
    "next_nofin_linear/cost": (-0.012928020998947442, -0.012930510086637419),
    "next_nofin_linear/total": (-0.012314429022478044, -0.011883059703965543),
    "walk_forward/carry": (0.0028612307597038027, 0.0028476105375228773),
    "walk_forward/cost": (-0.0005972340491412645, -0.0005957743190115672),
    "walk_forward/total": (0.11050304329894778, 0.1099870381166468),
}



def _flat(obj, prefix=""):
    out = {}
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            out.update(_flat(obj[c], f"{prefix}{c}/"))
    elif isinstance(obj, pd.Series):
        out[prefix.rstrip("/")] = [float(v) for v in obj.to_numpy(dtype=float)]
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}{k}/"))
    elif isinstance(obj, (int, float, np.floating, np.integer)):
        out[prefix.rstrip("/")] = [float(obj)]
    return out


def _check(cases):
    got = {}
    for name, obj in cases.items():
        got.update(_flat(obj, f"{name}/"))
    if os.environ.get("QP_REGEN_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(got, indent=0, sort_keys=True))
    want = json.loads(GOLDEN.read_text())
    assert sorted(got) == sorted(want)
    for k in want:
        np.testing.assert_allclose(got[k], want[k], rtol=0, atol=1e-12, err_msg=k)
    before = json.loads(BEFORE.read_text())
    assert sorted(before) == sorted(want)
    for k, (b, a) in BEFORE_AFTER.items():
        assert abs(sum(before[k]) - b) <= 1e-10, k
        assert abs(sum(got[k]) - a) <= 1e-10, k


def _market(n=400, seed=11):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 0.004, n))
    p2 = pd.Series(1.30 * np.exp(common), index=idx)
    p1 = pd.Series(1.10 * np.exp(0.9 * common + 0.02 * np.sin(np.arange(n) / 9)
                                 + rng.normal(0, 0.002, n)), index=idx)
    pos = np.sign(np.sin(np.arange(n) / 13)) * (np.arange(n) % 5 != 0)
    rates = {"rb1": 0.02, "rq1": 0.05, "rb2": pd.Series(0.01 + 0.001 * np.cos(np.arange(n) / 50), index=idx),
             "rq2": 0.045}
    return p1, p2, pos.astype(float), rates


def _res(r):
    return {"spot": r.spot_pnl, "carry": r.carry_pnl, "cost": r.cost_pnl, "total": r.total_pnl,
            "leg1": r.leg1_pnl, "leg2": r.leg2_pnl, "fin": r.financing_pnl, "rf": r.risk_free_pnl}


def test_fx_pairs_characterisation():
    p1, p2, pos, rates = _market()
    fin = Financing(rate=0.04, lend_spread_bp=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cases = {
            "next_fin": _res(run_backtest(p1, p2, pos, 0.85, pair1="EURUSD", pair2="GBPUSD", pip_spread_1=0.8,
                                          pip_spread_2=1.2, rates=rates, notional=1e6, financing=fin,
                                          capital=2e5)),
            "next_nofin_linear": _res(run_backtest(p1, p2, pos, 0.85, rates=rates, carry_method="linear",
                                                   execution=ExecutionConfig(fill_fraction=0.25))),
            "auction": _res(run_backtest(p1, p2, pos, 0.85, rates=rates, financing=fin,
                                         execution=ExecutionConfig(mode="close_auction", auction_cost_bp=0.7))),
            "walk_forward": _res(walk_forward_backtest(p1, p2, formation=120, trading=40, require_coint=False,
                                                       rates=rates, financing=fin).result),
        }
    _check(cases)
