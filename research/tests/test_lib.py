"""Tests for research/lib (W6): multiplicity, universe, excess returns, costs,
diagnostics, panel tools and the power gate. Hand examples are worked in the
module docstrings; each test names the one it checks."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "research" / "lib"), str(REPO / "research" / "studies"),
                str(REPO / "research" / "tools"), str(REPO / "data" / "fetch")]

import costs  # noqa: E402
import diagnostics as dg  # noqa: E402
import excess_returns as xr  # noqa: E402
import multiplicity  # noqa: E402
import panel  # noqa: E402
import power_gate as pg  # noqa: E402
from qp_core.stats import deflated_sharpe_detail  # noqa: E402
from universe import Universe, UniverseViolation  # noqa: E402

# ---- multiplicity ---------------------------------------------------------

def _log(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "trials.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_programme_tally_worked_example(tmp_path):
    rows = [
        {"kind": "train", "prereg": "a", "params": {"x": 1}, "metric": {"trial_count": 3, "sharpe": 0.2}},
        {"kind": "train", "prereg": "a", "params": {"x": 1}, "metric": {"trial_count": 3, "sharpe": 0.2}},
        {"kind": "train", "prereg": "b", "params": {}, "metric": {"trial_count": 1}},
        {"kind": "train", "prereg": "c", "params": {"y": 2}, "metric": {"trial_count": 7, "sharpe": -0.1}},
        {"kind": "withdrawn", "prereg": "d"},
        {"kind": "stress", "prereg": "c", "metric": {"sharpe_1bp_base": -0.3}},
    ]
    t = multiplicity.programme_tally(_log(tmp_path, rows))
    assert t.n_trials == 3 + 1 + 7 + 1  # duplicate rerun counted once
    per = np.array([0.2, -0.1, -0.3]) / math.sqrt(252)
    assert t.trial_sr_variance == pytest.approx(per.var(ddof=1), rel=1e-12)


def test_committed_log_counts_more_than_the_old_hard_coded_three():
    t = multiplicity.programme_tally()
    assert t.n_trials >= 13 and t.trial_sr_variance is not None


def test_dsr_hand_example_with_dispersed_trial_sharpes():
    """Acceptance: DSR against a hand example with dispersed trial Sharpes.
    N = 10, V = 0.0004 (per-period trial Sharpes with sd 0.02):
    E[max] = 0.02 ((1-g) Phi^-1(0.9) + g Phi^-1(1 - 1/(10e))) = 0.02 * 1.5746 = 0.031492."""
    g = 0.5772156649015329
    from scipy.stats import norm
    sr0 = 0.02 * ((1 - g) * norm.ppf(0.9) + g * norm.ppf(1 - 1 / (10 * math.e)))
    assert sr0 == pytest.approx(0.031492, abs=5e-6)
    rng = np.random.default_rng(11)
    r = rng.standard_normal(1000)
    r = (r - r.mean()) / r.std(ddof=1) + 0.06
    d = deflated_sharpe_detail(r, 10, trial_sr_variance=0.0004, se_method="psr")
    z = (0.06 - sr0) * math.sqrt(999) / math.sqrt(
        1 - d["skew"] * 0.06 + (d["kurtosis"] - 1) / 4 * 0.06**2)
    assert d["sr0_per_period"] == pytest.approx(sr0, rel=1e-12)
    assert d["dsr"] == pytest.approx(norm.cdf(z), rel=1e-12)


# ---- universe -------------------------------------------------------------

def test_universe_worked_example_and_pegs():
    u = Universe.load()
    m = u.members("2008-06-30")
    assert "SKK" in m and "SIT" not in m and "USD" in m
    assert "DKK" not in u.members("2008-06-30", include_pegs=False)
    assert "RUB" not in u.members("2022-06-01")  # 'auto' first date unresolved without data too
    assert "HRK" not in u.members("2023-01-02")


def test_universe_has_thirty_plus_currencies_with_sources():
    t = pd.read_csv(REPO / "research" / "universe" / "fx.csv", dtype=str, keep_default_na=False)
    assert len(t) >= 30
    assert (t["reason"] != "").all() and (t["source"] != "").all()
    assert set(t["regime"]) <= {"float", "peg"}


def test_study_holding_a_dead_currency_fails():
    """Acceptance: a test fails if a study uses a currency outside its live window."""
    u = Universe.load()
    idx = pd.bdate_range("2014-12-30", "2015-01-05")
    w = pd.DataFrame({"LTL": 0.5, "USD": -0.5}, index=idx)
    with pytest.raises(UniverseViolation, match="LTL"):
        u.assert_live(w)
    w.loc[w.index >= "2015-01-01", "LTL"] = 0.0
    u.assert_live(w)


def test_auto_first_date_comes_from_the_data():
    idx = pd.bdate_range("2005-03-30", periods=5)
    px = pd.DataFrame({"CNY": [np.nan, np.nan, 10.0, 10.1, 10.2]}, index=idx)
    u = Universe.load(px)
    assert u.table.loc["CNY", "first"] == idx[2]


# ---- excess returns and publication lags ---------------------------------

def test_excess_return_two_day_hand_example():
    idx = pd.DatetimeIndex(["2024-03-04", "2024-03-05"])
    spot = pd.Series([1.0, 1.01], index=idx)
    out = xr.excess_returns(spot, pd.Series(0.05, index=idx), pd.Series(0.01, index=idx),
                            foreign="EUR", base="USD")
    assert out["excess"].iloc[1] == pytest.approx(math.log(1.01) + 0.04 / 360, rel=1e-12)
    assert out["excess"].iloc[1] == pytest.approx(0.01006144, abs=5e-9)


def test_weekend_carry_accrues_three_days():
    idx = pd.DatetimeIndex(["2024-03-08", "2024-03-11"])  # Friday, Monday
    out = xr.excess_returns(pd.Series([1.0, 1.0], index=idx), pd.Series(0.05, index=idx),
                            pd.Series(0.0, index=idx), "GBP", "USD")
    assert out["days"].iloc[1] == 3
    assert out["carry"].iloc[1] == pytest.approx(0.05 * 3 / 365)


def test_monthly_rate_is_not_used_before_month_m_plus_2():
    """Acceptance: a test fails if a study uses a rate before its release date."""
    monthly = pd.Series([0.01, 0.02], index=pd.DatetimeIndex(["2024-01-31", "2024-02-29"]))
    dates = pd.bdate_range("2024-02-26", "2024-04-05")
    known = xr.as_known(monthly, "monthly", dates)
    assert known.loc[:"2024-02-29"].isna().all()
    assert (known.loc["2024-03-01":"2024-03-29"] == 0.01).all()
    assert known.loc["2024-04-01"] == 0.02
    naive = monthly.reindex(dates, method="ffill")  # the old in-month use
    with pytest.raises(xr.LookAheadViolation):
        xr.assert_known(naive.dropna(), monthly, "monthly")
    xr.assert_known(known, monthly, "monthly")


def test_daily_fixing_known_next_business_day():
    d = pd.Series([0.039, 0.040], index=pd.DatetimeIndex(["2024-03-08", "2024-03-11"]))
    av = xr.available_from(pd.DatetimeIndex(d.index), "daily")
    assert list(av) == [pd.Timestamp("2024-03-11"), pd.Timestamp("2024-03-12")]


def test_invert_ecb():
    px = pd.DataFrame({"USD": [1.10], "GBP": [0.85]})
    inv = xr.invert_ecb(px)
    assert inv["GBP"].iloc[0] == pytest.approx(1.10 / 0.85) and inv["EUR"].iloc[0] == 1.10


# ---- costs ----------------------------------------------------------------

def test_cost_tiers_stress_and_grid():
    idx = pd.bdate_range("2024-01-01", periods=60)
    rng = np.random.default_rng(0)
    lr = pd.DataFrame({"MXN": rng.normal(0, 0.01, 60)}, index=idx)
    lr.iloc[-1] = 0.2  # a stress day
    to = pd.DataFrame({"MXN": 1.0}, index=idx)
    calm = costs.daily_cost(to)
    assert calm.iloc[0] == pytest.approx(5e-4)
    stressed = costs.daily_cost(to, lr)
    assert stressed.iloc[-1] > stressed.iloc[30] >= 5e-4
    assert stressed.iloc[-1] <= 1e-3 + 1e-15
    g = costs.cost_grid(pd.Series(2e-4, index=idx), pd.Series(1e-4, index=idx))
    assert g["mean_net_0x"] == pytest.approx(2e-4) and g["mean_net_2x"] == pytest.approx(0.0)
    assert g["breakeven_multiplier"] == pytest.approx(2.0)
    assert costs.tier_bp("XYZ") == costs.DEFAULT_BP


# ---- diagnostics ----------------------------------------------------------

def test_pairs_numeraire_two_day_hand_example():
    """Acceptance: the pairs numeraire on a two-day hand example (EUR)."""
    idx = pd.bdate_range("2024-01-02", periods=2)
    usd = dg.leg_pnl_eur(pd.Series([1.10, 1.09], index=idx), 100.0)
    gbp = dg.leg_pnl_eur(pd.Series([0.85, 0.86], index=idx), -100.0)
    assert usd.iloc[1] == pytest.approx(0.917431, abs=1e-6)
    assert gbp.iloc[1] == pytest.approx(1.162791, abs=1e-6)
    assert (usd + gbp).iloc[1] == pytest.approx(2.080222, abs=1e-6)


def test_concentration_flags_an_outlier_series():
    idx = pd.bdate_range("2014-01-01", "2016-12-30")
    r = pd.Series(1e-5, index=idx)
    r.loc["2015-01-15"] = 0.5
    c = dg.concentration(r)
    assert c["top_k_share"]["5"] > 0.9 and c["kurtosis_gate"] == "fail"
    assert c["pnl_by_year"]["2015"] > c["pnl_by_year"]["2014"]
    assert c["sharpe_without_year"] is None or math.isfinite(c["sharpe_without_year"])


def test_fill_lag_shifts_one_day():
    s = pd.Series([0.0, 1.0, 1.0, 0.0])
    assert list(dg.lag_fill(s)) == [0.0, 0.0, 1.0, 1.0]


# ---- panel ----------------------------------------------------------------

def test_lrv_sort_worked_example():
    sig = pd.DataFrame([[1, 2, 3, 4, 5, 6]], columns=list("ABCDEF"), dtype=float)
    ret = sig / 100.0
    out = panel.lrv_sort(sig, ret, n_port=2)
    assert out["HML"].iloc[0] == pytest.approx(0.03)
    live = pd.DataFrame([[True] * 5 + [False]], columns=list("ABCDEF"))
    assert panel.lrv_sort(sig, ret, n_port=2, live=live)["HML"].iloc[0] == pytest.approx(
        (4 + 5) / 200 - (1 + 2 + 3) / 300)


def test_fama_macbeth_recovers_a_planted_slope():
    rng = np.random.default_rng(5)
    months, n = 240, 30
    sig = pd.DataFrame(rng.standard_normal((months, n)))
    ret = 0.002 * sig + pd.DataFrame(rng.normal(0, 0.03, (months, n)))
    fm = panel.fama_macbeth(sig, ret)
    assert fm["t"] > 3 and fm["mean"] == pytest.approx(0.002, abs=0.001)
    fm0 = panel.fama_macbeth(sig, pd.DataFrame(rng.normal(0, 0.03, (months, n))))
    assert abs(fm0["t"]) < 3


def test_correlation_check_pass_and_fail():
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0, 0.02, 120))
    ok = panel.correlation_check(a + rng.normal(0, 0.003, 120), a)
    assert ok["pass"] and ok["corr"] > 0.8
    assert not panel.correlation_check(pd.Series(rng.normal(0, 0.02, 120)), a)["pass"]


# ---- power gate -----------------------------------------------------------

def test_power_gate_blocks_the_13_percent_fixture():
    """Acceptance: the gate blocks a 13%-power fixture (the signed momentum file)."""
    res = pg.gate_from_simulation(REPO / "research" / "power" / "momentum-fx-g10.json", 0.06)
    assert res.power == pytest.approx(0.13) and res.verdict == "blocked"
    assert pg.gate("single_series").verdict == "monitoring"
    assert pg.analytic_power(0.3, 2.0) == pytest.approx(0.111, abs=5e-4)


def test_power_gate_passes_a_panel_fixture():
    """Acceptance: the gate passes a panel fixture (docstring worked examples)."""
    weak = pg.gate("panel_fama_macbeth", n_currencies=30, ic_monthly=0.04, n_eff=15,
                   holdout_years=10.9)
    assert weak.verdict == "blocked" and weak.power == pytest.approx(0.551, abs=5e-4)
    strong = pg.gate("panel_fama_macbeth", n_currencies=30, ic_monthly=0.06, n_eff=15,
                     holdout_years=10.9)
    assert strong.verdict == "confirmatory" and strong.power == pytest.approx(0.844, abs=5e-4)
    assert pg.gate("panel_fama_macbeth", n_currencies=10, ic_monthly=0.06, n_eff=15,
                   holdout_years=10.9).verdict == "blocked"


def test_panel_power_formula_matches_simulation():
    """The analytic panel power agrees with a Monte Carlo of the FM t-test
    (400 reps, binomial SE about 0.025; tolerance 3 SE)."""
    rng = np.random.default_rng(9)
    months, n, ic = 60, 30, 0.06
    sr_ann = pg.panel_slope_sharpe(ic, n)
    expect = pg.analytic_power(sr_ann, months / 12)
    hits = 0
    for _ in range(400):
        x = rng.standard_normal((months, n))
        y = ic * x + math.sqrt(1 - ic**2) * rng.standard_normal((months, n))
        fm = panel.fama_macbeth(pd.DataFrame(x), pd.DataFrame(y))
        hits += fm["t"] > 1.645
    assert hits / 400 == pytest.approx(expect, abs=3 * math.sqrt(expect * (1 - expect) / 400))
