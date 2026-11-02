"""Tests for the forward-hold-out registrations' training code. No network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "research" / "tools"), str(REPO / "research" / "studies"), str(REPO / "data" / "fetch")]

import forward_studies as fs  # noqa: E402
import fx_studies  # noqa: E402
import prereg  # noqa: E402
from models import HoldoutViolation  # noqa: E402

CCY = ["AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NOK", "NZD", "SEK"]


def _px(n: int = 400, seed: int = 1, start: str = "2010-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.006, (n, len(CCY))), axis=0)), index=idx, columns=CCY)


# ---- registrations and committed results ---------------------------------

@pytest.mark.parametrize("name", ["pairs-fx-g10-v2.md", "momentum-fx-g10.md"])
def test_new_registrations_are_signed(name):
    reg = prereg.load(REPO / "research" / "prereg" / name)
    assert reg.signed and reg.signed_on == "2026-11-02"


def test_training_results_match_registrations_and_stay_in_training():
    for res_name, reg_name in [("pairs_fx_g10_v2_train.json", "pairs-fx-g10-v2.md"),
                               ("momentum_fx_g10_train.json", "momentum-fx-g10.md")]:
        path = REPO / "research" / "results" / res_name
        res = json.loads(path.read_text(encoding="utf-8"))
        assert res["prereg"] == f"research/prereg/{reg_name}"
        assert res["period"]["end"] <= "2021-12-31"
        assert prereg.check_result(path) == []


def test_momentum_parameters_match_registration_text():
    text = (REPO / "research" / "prereg" / "momentum-fx-g10.md").read_text(encoding="utf-8")
    p = fs.MOMENTUM
    assert f"previous {p['lookback_days']} fixes" in text
    assert "Long the 3 highest scores, short the 3 lowest" in text and p["n_long"] == p["n_short"] == 3
    assert f"{p['cost_bp']:g} bp per unit of gross weight" in text
    assert f"Trial\ncount: {p['trial_count']}." in text or f"Trial count: {p['trial_count']}" in text


def test_pairs_v2_text_matches_code_and_result():
    text = (REPO / "research" / "prereg" / "pairs-fx-g10-v2.md").read_text(encoding="utf-8")
    res = json.loads((REPO / "research" / "results" / "pairs_fx_g10_v2_train.json").read_text(encoding="utf-8"))
    assert f"Trial count {fs.PAIRS_V2_TRIALS}" in text and res["trial_count"] == fs.PAIRS_V2_TRIALS
    assert f"net annualised Sharpe {res['sharpe']:.3f}" in text
    assert f"trades in {res['years_with_trades']} of 19 years" in text
    assert set(fs.VAR_RULES) == set(res["superseded_kalman_menu"]["variants"])
    for k in ("delta", "entry_z", "exit_z", "cost_bp", "embargo_days"):
        assert fs.PAIRS_V2[k] == fx_studies.PAIRS[k]


def test_power_table_quotes_the_power_file():
    text = (REPO / "research" / "prereg" / "pairs-fx-g10-v2.md").read_text(encoding="utf-8")
    doc = json.loads((REPO / "research" / "power" / "pairs_v2_power.json").read_text(encoding="utf-8"))
    rows = [ln for ln in text.splitlines() if ln.startswith("| half-life") or ln.startswith("| no coint")]
    assert len(rows) == len(doc["cells"]) == 6
    for row, c in zip(rows, doc["cells"], strict=True):
        cols = [x.strip() for x in row.strip("|").split("|")]
        assert cols[2] == f"{100 * c['selection_rate']:.1f} %"
        assert float(cols[3].rstrip(" %")) == pytest.approx(100 * c["level"]["rejection_rate"], abs=0.051)
        assert float(cols[5].rstrip(" %")) == pytest.approx(100 * c["kalman_v1"]["rejection_rate"], abs=0.051)


# ---- training guard -------------------------------------------------------

def test_studies_refuse_holdout_dates():
    late = _px(300, start="2021-06-01")
    assert late.index.max() >= pd.Timestamp("2022-01-01")
    with pytest.raises(HoldoutViolation):
        fs.momentum_study(late)
    with pytest.raises(HoldoutViolation):
        fs.pairs_v2_study(late)


# ---- momentum -------------------------------------------------------------

def test_momentum_ranks_and_unit_gross():
    idx = pd.bdate_range("2010-01-01", periods=130)
    drift = np.arange(len(CCY)) * 1e-3  # SEK strongest, AUD weakest
    px = pd.DataFrame(np.exp(np.outer(np.arange(len(idx)), drift)), index=idx, columns=CCY)
    w = fs.momentum_weights(px, dict(fs.MOMENTUM, lookback_days=20))
    last = w.iloc[-1]
    assert np.isclose(last.abs().sum(), 1.0) and np.isclose(last.sum(), 0.0)
    assert set(last[last > 0].index) == {"NOK", "NZD", "SEK"}
    assert set(last[last < 0].index) == {"AUD", "CAD", "CHF"}
    assert (w.iloc[:20] == 0).all().all()  # no weights before a full lookback


def test_momentum_rebalances_only_at_month_end():
    px = _px(300)
    w = fs.momentum_weights(px)
    idx = pd.DatetimeIndex(w.index)
    month_end = np.r_[idx.month[1:] != idx.month[:-1], True]
    changed = (w.diff().abs().sum(axis=1) > 0).to_numpy().copy()
    changed[0] = False
    assert not (changed & ~month_end).any()
    assert changed.any()


def test_momentum_ties_go_to_alphabetical_order():
    idx = pd.bdate_range("2010-01-01", periods=80)
    px = pd.DataFrame(1.0, index=idx, columns=CCY[::-1])  # flat: every score ties
    last = fs.momentum_weights(px, dict(fs.MOMENTUM, lookback_days=10)).iloc[-1]
    assert list(last.index) == sorted(CCY)
    assert set(last[last < 0].index) == {"AUD", "CAD", "CHF"}
    assert set(last[last > 0].index) == {"NOK", "NZD", "SEK"}


def test_momentum_return_is_next_day_and_costed():
    px = _px(200)
    p = dict(fs.MOMENTUM, cost_bp=0.0)
    w = fs.momentum_weights(px, p)
    r = np.log(px[w.columns]).diff()
    t = 150
    expected = float((w.iloc[t - 1] * r.iloc[t]).sum())
    assert np.isclose(fs.momentum_returns(px, p).loc[px.index[t]], expected)
    costed = fs.momentum_returns(px, dict(fs.MOMENTUM, cost_bp=10.0))
    traded = w.diff().abs().sum(axis=1).iloc[1:]
    assert np.allclose(fs.momentum_returns(px, p) - costed, 10e-4 * traded)


def test_momentum_no_lookahead():
    px = _px(300)
    base = fs.momentum_returns(px)
    cut = px.index[200]
    bumped = px.copy()
    bumped.loc[bumped.index > cut] *= np.linspace(1.0, 2.0, int((bumped.index > cut).sum()))[:, None] ** \
        np.arange(1, len(CCY) + 1)
    other = fs.momentum_returns(bumped)
    assert np.allclose(base.loc[:cut], other.loc[:cut])


# ---- pairs v2 -------------------------------------------------------------

def _pair(n: int = 900, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lx = np.cumsum(rng.normal(0, 0.01, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.95 * spread[t - 1] + rng.normal(0, 0.004)
    ly = 0.2 + 0.8 * lx + spread
    test = np.zeros(n, bool)
    test[650:] = True
    return ly, lx, test


def test_levels_rule_reproduces_the_signed_design():
    ly, lx, test = _pair()
    mine, _ = fs.pair_returns_rule(ly, lx, test, "levels")
    ref = fx_studies.pair_returns(pd.Series(ly), pd.Series(lx), test, fx_studies.PAIRS)
    assert np.allclose(mine, ref, rtol=0, atol=1e-15)


def test_smaller_variance_rules_trade_more():
    ly, lx, test = _pair()
    n_levels = fs.pair_returns_rule(ly, lx, test, "levels")[1]
    n_tenth = fs.pair_returns_rule(ly, lx, test, "levels_x0.1")[1]
    assert n_tenth >= n_levels
    r, _ = fs.pair_returns_rule(ly, lx, test, "diff")
    assert r.shape == (int(test.sum()),) and np.isfinite(r).all()


def test_flat_residual_is_refused():
    lx = np.zeros(300)
    ly = np.full(300, 0.5)  # the OLS residual is exactly zero
    test = np.zeros(300, bool)
    test[250:] = True
    with pytest.raises(ValueError):
        fs.pair_returns_rule(ly, lx, test, "diff")


def test_level_z_ignores_price_level():
    ly, lx, test = _pair()
    r0, a0 = fs.pair_returns_level(ly, lx, test)
    r1, a1 = fs.pair_returns_level(ly, lx + np.log(0.0067), test)  # JPY-sized quote for x
    assert np.allclose(r0, r1, rtol=0, atol=1e-13)
    assert a0.pop("max_abs_z") == pytest.approx(a1.pop("max_abs_z"), rel=1e-10) and a0 == a1


def test_level_z_by_hand():
    ly, lx, test = _pair()
    form = ~test
    b, a = np.polyfit(lx[form], ly[form], 1)
    z = (ly - a - b * lx) / np.std(ly[form] - a - b * lx[form])
    _, act = fs.pair_returns_level(ly, lx, test)
    assert act["entry_signal_days"] == int((np.abs(z[test]) > 2.0).sum())
    assert act["max_abs_z"] == pytest.approx(np.abs(z[test]).max(), rel=1e-9)


def test_level_z_trades_where_kalman_does_not():
    ly, lx, test = _pair()
    _, lev = fs.pair_returns_level(ly, lx, test)
    kal: dict = {}
    fx_studies.pair_returns(pd.Series(ly), pd.Series(lx), test, fx_studies.PAIRS, kal)
    assert lev["entries"] > kal["entries"]


def test_level_z_no_lookahead():
    ly, lx, test = _pair()
    base, _ = fs.pair_returns_level(ly, lx, test)
    ly2 = ly.copy()
    ly2[800:] += 0.5
    other, _ = fs.pair_returns_level(ly2, lx, test)
    k = 800 - 650  # test rows before the change (return of row 800 uses its price)
    assert np.allclose(base[:k], other[:k])


def test_level_z_zero_residual_refused():
    with pytest.raises(ValueError):
        fs.pair_returns_level(np.full(300, 0.5), np.zeros(300), np.r_[np.zeros(250, bool), np.ones(50, bool)])


def test_power_simulation_smoke():
    import pairs_power as pp
    cell = pp.run_cell(10, pp.LEVELS["jpy_like"], reps=3, n_pairs=2, seed=5)
    assert 0.0 <= cell["selection_rate"] <= 1.0
    assert cell["level"]["share_test_days_abs_z_above_2"] > cell["kalman_v1"]["share_test_days_abs_z_above_2"]


def test_choose_rule_applies_the_activity_floor():
    v = {"levels": {"sharpe": 0.30, "years_with_trades": 3},
         "levels_x0.25": {"sharpe": 0.10, "years_with_trades": 15},
         "levels_x0.1": {"sharpe": None, "years_with_trades": 0},
         "diff": {"sharpe": 0.20, "years_with_trades": 17}}
    assert fs.choose_rule(v, 19, 0.0) == "levels"
    assert fs.choose_rule(v, 19, 0.75) == "diff"
    with pytest.raises(ValueError):
        fs.choose_rule(v, 19, 0.95)
