"""W24b characterisation corpus: the current backtester output, captured
before the move onto the qp_bt core. After migration every value must match
to 1e-12 of each series' scale, or the difference is listed in MIGRATION.md and checked against a
hand fixture (NEXT_RELEASE_PLAN W24 acceptance). Regenerate only on purpose:
QP_REGEN_GOLDEN=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fx_algo import BacktestConfig, ExecutionConfig, IntradayBacktester

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
        # 1e-12 of the series' scale: the pip columns are quote P&L / pip_size,
        # so floating-point reordering in the qp_bt engine shows at ~1e-12 pips
        scale = max(1.0, float(np.nanmax(np.abs(want[k]))) if want[k] else 1.0)
        np.testing.assert_allclose(got[k], want[k], rtol=0, atol=1e-12 * scale, err_msg=k)


SPREADS = {"asia": 0.6, "london": 0.4, "overlap": 0.2, "ny": 0.4, "late": 1.0}


def _bars(n=300, seed=3):
    rng = np.random.default_rng(seed)
    hours = np.arange(n, dtype=float) + 1.0
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    bars = pd.DataFrame({"close": close, "hour": np.mod(hours, 24.0)},
                        index=pd.Index(hours, name="bar_end_hours"))
    pos = pd.Series(np.round(np.sin(hours / 7) * 3) / 3, index=bars.index)
    return bars, pos


def test_fx_algo_characterisation():
    bars, pos = _bars()
    cfg = BacktestConfig(pip_size=1e-4, spread_pips_by_session=SPREADS, r_base=0.035, r_quote=0.05)
    bt = IntradayBacktester(cfg)
    cases = {}
    for name, ex in {"next": None, "next_f25": ExecutionConfig(fill_fraction=0.25),
                     "auction": ExecutionConfig(mode="close_auction", auction_cost_bp=0.5)}.items():
        ledger, summary = bt.run(bars, pos, execution=ex)
        num = ledger.select_dtypes("number")
        cases[name] = {"ledger": num, "summary": {k: v for k, v in summary.items()
                                                   if isinstance(v, (int, float, np.floating, np.integer))}}
    _check(cases)
