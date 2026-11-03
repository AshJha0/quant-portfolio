"""Historical VaR ``rule=``: both rules against the mpmath oracle (W5 values)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from eq_var.historical_var import historical_var

ORACLE = json.loads((Path(__file__).resolve().parent / "golden" / "oracle_vectors.json").read_text())
V = ORACLE["values"]


def sample() -> np.ndarray:
    return np.array([math.floor(1000.0 * math.sin(0.37 * i + 0.1) * (1.0 + 0.5 * math.cos(0.11 * i))
                                * 100.0 + 0.5) / 100.0 for i in range(250)])


@pytest.mark.parametrize("a_s", ORACLE["inputs"]["hist_alphas"])
def test_both_rules_match_oracle(a_s: str) -> None:
    a, t = float(a_s), a_s.replace(".", "")
    p = sample()
    assert historical_var(p, a) == V[f"hist_kth_var_a{t}"]
    assert historical_var(p, a, rule="kth_worst") == V[f"hist_kth_var_a{t}"]
    got = historical_var(p, a, rule="interpolated")
    assert got == pytest.approx(V[f"hist_interp_var_a{t}"], rel=1e-12, abs=0.0)


def test_interpolated_equals_numpy_type7() -> None:
    p = sample()
    for a in (0.004, 0.01, 0.025, 0.05, 0.1):
        assert historical_var(p, a, rule="interpolated") == pytest.approx(
            float(np.quantile(-p, 1.0 - a)), rel=1e-12)


def test_mar99_99pct_on_250_days() -> None:
    p = sample()
    losses = np.sort(-p)[::-1]
    assert historical_var(p, 0.01) == losses[2]
    v = historical_var(p, 0.01, rule="interpolated")
    assert v == pytest.approx(losses[2] + 0.49 * (losses[3] - losses[2]), rel=1e-14)


def test_unknown_rule_raises() -> None:
    with pytest.raises(ValueError, match="rule must be"):
        historical_var(sample(), 0.01, rule="type7")
