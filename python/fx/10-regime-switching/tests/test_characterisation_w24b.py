"""W24b characterisation corpus: the current backtester output, captured
before the move onto the qp_bt core. After migration every value must match
to 1e-12, or the difference is listed in MIGRATION.md and checked against a
hand fixture (NEXT_RELEASE_PLAN W24 acceptance). Regenerate only on purpose:
QP_REGEN_GOLDEN=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fx_regime import ExecutionConfig, StrategyConfig, generate_roro_panel, run_backtest, static_carry_regimes

GOLDEN = Path(__file__).parent / "golden" / "characterisation_w24b.json"


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
