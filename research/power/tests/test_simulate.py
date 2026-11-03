"""Checks for research/power/simulate.py and the three Round 18 power files."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(REPO / "research" / "power"),
    str(REPO / "research" / "tools"),
    str(REPO / "research" / "lib"),
]

import power_gate as pg  # noqa: E402
import prereg  # noqa: E402
import simulate as sim  # noqa: E402

FILES = ("combination-panel.json", "ust-slope-fly.json", "tsmom-cross-asset.json")


def test_panel_strong_effect_on_independent_currencies_is_found():
    # IC 0.10, N 30 independent: analytic power Phi(0.10 sqrt(30) sqrt(12) sqrt(10.83) - 1.645) ~ 1.
    assert pg.analytic_power(pg.panel_slope_sharpe(0.10, 30), 130 / 12) > 0.99
    rng = np.random.default_rng(5)
    t = [sim.combination_panel_once(rng, 0.10, rho_signal=0.0, rho_noise=0.0) for _ in range(30)]
    assert sim._rate(t) >= 0.95


def test_panel_null_rate_near_five_percent():
    rng = np.random.default_rng(6)
    t = [sim.combination_panel_once(rng, 0.0) for _ in range(150)]
    assert sim._rate(t) < 0.12  # 5 % plus three Monte Carlo standard errors


def test_ust_null_and_planted():
    rng = np.random.default_rng(7)
    assert sim._rate([sim.ust_once(rng, 0.0) for _ in range(300)]) < 0.10
    assert sim._rate([sim.ust_once(rng, 0.3) for _ in range(50)]) > 0.9


def test_tsmom_draw_returns_t_and_sharpe():
    t, sr = sim.tsmom_once(np.random.default_rng(8), 0.0)
    assert t is not None and sr is not None and math.isfinite(t)


@pytest.mark.parametrize("name", FILES)
def test_power_files_satisfy_the_signing_check(name):
    ref = f"research/power/{name}"
    assert prereg.power_errors(f"Power: {ref}\n## Signature\n", REPO) == []
    doc = json.loads((REPO / ref).read_text(encoding="utf-8"))
    null = doc["cells"][0]
    assert null["truth"] == "null" and abs(null["rejection_rate"] - 0.05) < 3 * 0.011 + 0.01
    rates = [c["rejection_rate"] for c in doc["cells"]]
    assert rates[-1] > rates[1]
