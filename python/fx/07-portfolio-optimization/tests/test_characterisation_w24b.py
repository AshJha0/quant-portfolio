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
from fx_port import run_backtest

GOLDEN = Path(__file__).parent / "golden" / "characterisation_w24b.json"
BEFORE = Path(__file__).parent / "golden" / "characterisation_w24b_before.json"
# key: (sum before migration, sum after), filled from the corpus
BEFORE_AFTER: dict[str, tuple[float, float]] = {
    "a/ledger/carry_pnl": (-0.0025106026217990907, -0.002635134605532181),
    "a/ledger/cost": (0.0007163280739112415, 0.0007282437548880027),
    "a/ledger/net": (-0.04183837019222526, -0.03887369077416137),
    "a/ledger/turnover": (7.163280739112414, 7.282437548880378),
    "b/ledger/carry_pnl": (-0.006580465801792353, -0.006675252377918253),
    "b/ledger/cost": (0.001711875693357934, 0.0017703470238493724),
    "b/ledger/net": (-0.071556169339078, -0.08535754181391182),
    "b/ledger/turnover": (16.42033324085268, 16.705501417179487),
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
        assert abs(sum(before[k]) - b) <= 1e-12 * max(1.0, abs(b)) * len(before[k]), k  # relative: last-ulp libm differences across OSes
        assert abs(sum(got[k]) - a) <= 1e-12 * max(1.0, abs(a)) * len(got[k]), k


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
