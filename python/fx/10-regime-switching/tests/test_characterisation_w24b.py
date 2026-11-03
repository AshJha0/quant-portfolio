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
from pathlib import Path

import numpy as np
import pandas as pd
from fx_regime import ExecutionConfig, StrategyConfig, generate_roro_panel, run_backtest, static_carry_regimes

GOLDEN = Path(__file__).parent / "golden" / "characterisation_w24b.json"
BEFORE = Path(__file__).parent / "golden" / "characterisation_w24b_before.json"
# key: (sum before migration, sum after), filled from the corpus
BEFORE_AFTER: dict[str, tuple[float, float]] = {
    "static/ledger/carry": (0.1055363695951425, 0.10516165412349429),
    "static/ledger/cost": (0.0029993263224677405, 0.0031065291205613077),
    "static/ledger/net": (0.0699743903064405, 0.07490397111623362),
    "static/ledger/spot": (-0.03256265296623427, -0.027151153886699367),
    "static/ledger/turnover": (7.7061588344582965, 7.986530247584562),
    "switch/ledger/carry": (0.01059976231173355, 0.010330043991874517),
    "switch/ledger/cost": (0.008228634595589288, 0.00836319420266507),
    "switch/ledger/net": (-0.03885914636531946, -0.032881167409794465),
    "switch/ledger/spot": (-0.04123027408146373, -0.034848017199003906),
    "switch/ledger/turnover": (26.07048671098978, 26.43717628962606),
    "switch_auction/ledger/carry": (0.010725465516769126, 0.010570811794275495),
    "switch_auction/ledger/cost": (0.03429912130657907, 0.034845043476376514),
    "switch_auction/ledger/net": (-0.1036759299916523, -0.09842065266325328),
    "switch_auction/ledger/spot": (-0.08010227420184235, -0.07414642098115226),
    "switch_auction/ledger/turnover": (26.07048671098978, 26.46891241284428),
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
        # 1e-12 of the series' own scale: a P&L of 1e4 carries ~1e-12 relative
        # reordering noise between platforms (numpy reductions, libm), which is
        # far below anything a change in the backtester would produce.
        scale = max(1.0, float(np.max(np.abs(want[k]))) if len(want[k]) else 1.0)
        np.testing.assert_allclose(got[k], want[k], rtol=0, atol=1e-12 * scale, err_msg=k)
    before = json.loads(BEFORE.read_text())
    assert sorted(before) == sorted(want)
    for k, (b, a) in BEFORE_AFTER.items():
        assert abs(sum(before[k]) - b) <= 1e-10, k
        assert abs(sum(got[k]) - a) <= 1e-10, k


def test_fx_regime_characterisation():
    panel = generate_roro_panel(400, n_states=2, seed=42)
    idx = panel.returns.index
    labels = pd.Series(np.where((np.arange(len(idx)) // 60) % 2 == 0, "risk_on", "risk_off"), index=idx)
    cases = {}
    for name, (reg, ex) in {"static": (static_carry_regimes(idx), None),
                            "switch": (labels, None),
                            "switch_auction": (labels, ExecutionConfig(mode="close_auction"))}.items():
        r = run_backtest(panel.returns, panel.deposit_rates, reg, StrategyConfig(), execution=ex)
        cases[name] = {"ledger": r.ledger.select_dtypes("number"), "weights": r.weights}
    _check(cases)
