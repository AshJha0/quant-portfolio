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
from fx_port import run_backtest

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


def _panel(n=300, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    cols = ["AUD", "JPY", "EUR", "MXN"]
    spot = pd.DataFrame(rng.normal(0, 0.006, (n, 4)), index=idx, columns=cols)
    carry = pd.DataFrame(np.tile([0.03, -0.01, 0.0, 0.06], (n, 1)) / 252, index=idx, columns=cols)
    return spot, carry


def _inv_vol(hist):
    v = hist.std()
    return (1 / v) / (1 / v).sum() * np.sign(hist.mean())


def test_fx_port_characterisation():
    spot, carry = _panel()
    costs = pd.Series({"AUD": 1.0, "JPY": 0.5, "EUR": 0.3, "MXN": 4.0})
    cases = {}
    for name, kw in {"a": {"est_window": 60, "rebalance_every": 21, "cost_bps": 1.0},
                     "b": {"est_window": 30, "rebalance_every": 5, "cost_bps": costs}}.items():
        r = run_backtest(spot, carry, _inv_vol, **kw)
        cases[name] = {"ledger": r.ledger, "weights": r.weights}
    _check(cases)
