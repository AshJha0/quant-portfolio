"""Tests for research/combination (W23 part 2). Hand examples from the module docstrings."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "combination"),
                str(REPO / "python" / "fx" / "07-portfolio-optimization" / "src")]

import combine as cb  # noqa: E402
import panel_seam  # noqa: E402

M = pd.period_range("2000-01", periods=3, freq="M").to_timestamp("M")
CCY = ["AAA", "BBB", "CCC"]


def _frame(rows):
    return pd.DataFrame(rows, index=M[: len(rows)], columns=CCY, dtype=float)


def test_docstring_combination_and_vol_scale():
    carry = _frame([[-1, 0, 1]])
    value = _frame([[1, 0, -1]])
    mom = _frame([[-1, 0, 1]])
    c = cb.combine_signals({"carry": carry, "value": value, "momentum": mom})
    np.testing.assert_allclose(c.iloc[0].to_numpy(), [-1 / 3, 0, 1 / 3], atol=1e-15)
    w = cb.raw_weights(c.iloc[0]).to_numpy()
    np.testing.assert_allclose(w, [-0.5, 0, 0.5], atol=1e-15)
    spec = cb.CombinationSpec(target_vol=0.10, max_weight=10.0)
    out = cb.vol_target(w, np.eye(3) * 0.01, spec)
    k = 0.1 / math.sqrt(0.5 * 0.01 * 12)
    assert k == pytest.approx(0.408248290463863, abs=1e-14)
    np.testing.assert_allclose(out, w * k, atol=1e-15)


def test_limits_cap_gross_then_clip_name():
    spec = cb.CombinationSpec(target_vol=10.0, max_gross=2.0, max_weight=0.6)
    out = cb.vol_target(np.array([-0.5, 0.0, 0.5]), np.eye(3) * 1e-4, spec)
    np.testing.assert_allclose(out, [-0.6, 0.0, 0.6])  # gross 2 -> +-1, clipped to 0.6


def test_min_signals_and_live_mask():
    a = _frame([[1, 2, 3]])
    b = _frame([[3, np.nan, 1]])
    c = cb.combine_signals({"a": a, "b": b}, min_signals=2)
    assert math.isnan(c.iloc[0]["BBB"])
    live = pd.DataFrame([[True, True, False]], index=M[:1], columns=CCY)
    z = cb.cross_zscore(a, live)
    assert math.isnan(z.iloc[0]["CCC"])
    np.testing.assert_allclose(z.iloc[0][["AAA", "BBB"]].to_numpy(),
                               [-1 / math.sqrt(2), 1 / math.sqrt(2)])


def test_ewma_cov_matches_explicit_weights_and_shrink():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(size=(40, 3)) * 0.02)
    hl = 6.0
    lam = 0.5 ** (1 / hl)
    w = lam ** np.arange(39, -1, -1)
    w /= w.sum()
    want = sum(w[t] * np.outer(r.iloc[t], r.iloc[t]) for t in range(40))
    got = cb.ewma_cov(r, hl)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-18)
    s = cb.shrink_to_diag(got, 1.0)
    np.testing.assert_allclose(s, np.diag(np.diag(got)))


def test_run_no_lookahead_and_costs():
    """Weights at m depend only on rows < m of next_ret; changing the future
    return of month m must not change the weights set at m."""
    rng = np.random.default_rng(1)
    months = pd.period_range("2000-01", periods=60, freq="M").to_timestamp("M")
    ccy = [f"C{i}" for i in range(6)]
    sig = {k: pd.DataFrame(rng.normal(size=(60, 6)), index=months, columns=ccy)
           for k in ("carry", "value", "momentum")}
    nr = pd.DataFrame(rng.normal(size=(60, 6)) * 0.03, index=months, columns=ccy)
    spec = cb.CombinationSpec(min_history=24)
    costs = dict.fromkeys(ccy, 2.0)
    base = cb.run(sig, nr, costs, spec)
    nr2 = nr.copy()
    nr2.iloc[40] *= 50.0
    alt = cb.run(sig, nr2, costs, spec)
    pd.testing.assert_frame_equal(base.weights.iloc[:41], alt.weights.iloc[:41])
    assert not base.weights.iloc[41:].equals(alt.weights.iloc[41:])
    assert (base.weights.iloc[:24] == 0).all().all()
    np.testing.assert_allclose(base.cost, base.turnover * 2e-4, rtol=1e-14)
    np.testing.assert_allclose(base.net, base.gross - base.cost, rtol=1e-14)
    assert (base.weights.abs() <= spec.max_weight + 1e-15).all().all()
    vol = base.gross.iloc[24:].std() * math.sqrt(12)
    assert 0.05 < vol < 0.2


def test_cost_qp_sizer_direction_and_no_trade_band():
    size = cb.cost_qp_sizer(risk_aversion=10.0, alpha_scale=0.01)
    w = size(np.array([0.2, 0.0, -0.2]), np.eye(3) * 0.001, np.zeros(3), np.zeros(3))
    assert w[0] > 0 > w[2]
    w2 = size(np.array([0.2, 0.0, -0.2]), np.eye(3) * 0.001, np.zeros(3), np.full(3, 1.0))
    np.testing.assert_allclose(w2, 0.0, atol=1e-6)


def test_panel_seam():
    p = SimpleNamespace(carry=1, value=2, momentum=3, next_excess=4)
    got = panel_seam.from_object(p)
    assert got.signals == {"carry": 1, "value": 2, "momentum": 3} and got.live is None
    with pytest.raises(panel_seam.PanelUnavailable):
        panel_seam.from_object(SimpleNamespace(carry=1))


def test_spec_validation():
    with pytest.raises(ValueError):
        cb.CombinationSpec(shrink=1.5)
