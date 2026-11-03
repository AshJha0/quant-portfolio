"""Tests for research/panel (W23 part 1): the hold-out lock, the carry, value
and momentum signals, the panel runner and French factor momentum."""

from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "research" / "panel"), str(REPO / "research" / "lib"),
                str(REPO / "research" / "tools")]

import equity_factors as ef  # noqa: E402
import prereg  # noqa: E402
import signals as sg  # noqa: E402
import study  # noqa: E402
from holdout_lock import HoldoutLock, HoldoutLocked  # noqa: E402
from universe import Universe  # noqa: E402

MONTHS = pd.date_range("2000-01-31", "2015-12-31", freq="ME")

# ---- hold-out lock --------------------------------------------------------


def test_closed_lock_passes_2015_and_refuses_2016():
    lock = HoldoutLock()
    ok = pd.Series(1.0, index=pd.DatetimeIndex(["2015-12-31"]))
    assert lock.guard(ok) is ok
    with pytest.raises(HoldoutLocked):
        lock.guard(pd.Series(1.0, index=pd.DatetimeIndex(["2016-01-01"])))
    both = pd.Series(1.0, index=pd.DatetimeIndex(["2015-12-31", "2016-01-29"]))
    assert len(HoldoutLock.training(both)) == 1


REG = """# Test panel registration

Hold-out: 2016-01-01 to 2026-12-31.

## Signature

signed_by: {by}
signed_on: {on}
sha256: {sha}
"""


def _repo(tmp_path: Path, name: str, signed: bool) -> Path:
    d = tmp_path / "research" / "prereg"
    d.mkdir(parents=True)
    p = d / name
    p.write_text(REG.format(by="", on="", sha=""), encoding="utf-8", newline="\n")
    if signed:
        sha = prereg.registration_hash(p.read_text(encoding="utf-8"))
        p.write_text(REG.format(by="tester", on="2026-11-03", sha=sha), encoding="utf-8",
                     newline="\n")
    return p


def test_unsigned_or_undated_registration_cannot_open(tmp_path):
    p = _repo(tmp_path, "2026-11-03-panel.md", signed=False)
    with pytest.raises(HoldoutLocked, match="not signed"):
        HoldoutLock.open_with(p, repo=tmp_path, ledger=tmp_path / "l.jsonl")
    q = p.with_name("panel.md")
    p.rename(q)
    with pytest.raises(HoldoutLocked, match="dated"):
        HoldoutLock.open_with(q, repo=tmp_path, ledger=tmp_path / "l.jsonl")


def test_signed_registration_opens_once(tmp_path):
    p = _repo(tmp_path, "2026-11-03-panel.md", signed=True)
    ledger = tmp_path / "l.jsonl"
    now = dt.datetime(2026, 11, 3, tzinfo=dt.UTC)
    lock = HoldoutLock.open_with(p, repo=tmp_path, ledger=ledger, now=now)
    assert lock.is_open
    lock.guard(pd.Series(1.0, index=pd.DatetimeIndex(["2020-01-31"])))
    assert "2026-11-03T00:00:00+00:00" in ledger.read_text(encoding="utf-8")
    with pytest.raises(HoldoutLocked, match="already"):
        HoldoutLock.open_with(p, repo=tmp_path, ledger=ledger)


def test_editing_a_signed_registration_voids_it(tmp_path):
    p = _repo(tmp_path, "2026-11-03-panel.md", signed=True)
    p.write_text(p.read_text(encoding="utf-8").replace("Test panel", "Edited panel"),
                 encoding="utf-8", newline="\n")
    with pytest.raises(HoldoutLocked, match="not signed"):
        HoldoutLock.open_with(p, repo=tmp_path, ledger=tmp_path / "l.jsonl")


def test_registration_outside_prereg_dir_is_refused(tmp_path):
    p = tmp_path / "2026-11-03-panel.md"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(HoldoutLocked, match="not in"):
        HoldoutLock.open_with(p, repo=tmp_path, ledger=tmp_path / "l.jsonl")


# ---- signals ---------------------------------------------------------------


def test_worked_example_monthly_excess_return():
    idx = pd.DatetimeIndex(["2001-01-31", "2001-03-03"])  # 31 days
    spot = pd.DataFrame({"AUD": [1.00, 1.01]}, index=idx)
    rates = pd.DataFrame({"AUD": [0.05, 0.05], "USD": [0.01, 0.01]}, index=idx)
    rx = sg.monthly_excess_returns(spot, rates, "USD")
    # AUD is ACT/365 in the money market: carry = 0.05*31/365 - 0.01*31/360
    want = math.log(1.01) + 0.05 * 31 / 365 - 0.01 * 31 / 360
    assert rx["AUD"].iloc[1] == pytest.approx(want, abs=1e-15)
    spot2 = spot.rename(columns={"AUD": "EUR"})
    rates2 = rates.rename(columns={"AUD": "EUR"})
    rx2 = sg.monthly_excess_returns(spot2, rates2, "USD")
    assert rx2["EUR"].iloc[1] == pytest.approx(0.0133947, abs=1e-7)  # docstring figure


def test_carry_value_momentum_definitions():
    idx = MONTHS[:70]
    spot = pd.DataFrame({"A": np.linspace(1, 2, 70), "USD": 1.0}, index=idx)
    cpi = pd.DataFrame({"A": 100.0, "USD": 100.0}, index=idx)
    v = sg.value(spot, cpi, "USD", lookback=60)
    assert v["A"].iloc[65] == pytest.approx(-(math.log(spot["A"].iloc[65]) - math.log(spot["A"].iloc[5])))
    assert v["A"].iloc[:60].isna().all()
    rates = pd.DataFrame({"A": 0.05, "B": 0.02, "USD": 0.03}, index=idx)
    c = sg.carry(rates, "USD")
    assert c.iloc[0].to_dict() == pytest.approx({"A": 0.02, "B": -0.01})
    rx = pd.DataFrame({"A": np.arange(70.0)}, index=idx)
    m = sg.momentum(rx, lookback=12, skip=1)
    assert m["A"].iloc[12] == pytest.approx(sum(range(1, 12)))  # rows 1..11
    assert m["A"].iloc[11] == pytest.approx(sum(range(11)))  # rows 0..10
    assert np.isnan(m["A"].iloc[10])
    with pytest.raises(ValueError):
        sg.momentum(rx, lookback=3, skip=3)


def test_month_ends_takes_last_observation():
    d = pd.DataFrame({"x": [1.0, 2.0, 3.0]},
                     index=pd.DatetimeIndex(["2001-01-30", "2001-01-31", "2001-02-27"]))
    me = sg.month_ends(d)
    assert me["x"].to_list() == [2.0, 3.0]


# ---- panel runner ----------------------------------------------------------


def _toy_panel(noise: float, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cols = list("ABCDEF")
    sig = pd.DataFrame(rng.permuted(np.tile(np.arange(6.0), (len(MONTHS), 1)), axis=1),
                       index=MONTHS, columns=cols)
    rx = (sig.shift(1) / 100.0) + noise * rng.normal(size=sig.shape)
    live = pd.DataFrame(True, index=MONTHS, columns=cols)
    return rx, sig, live


def test_worked_example_rank_returns_give_three_percent():
    rx, sig, live = _toy_panel(0.0)
    res = study.run_panel(rx, {"carry": sig}, live, n_port=2)
    r = res["signals"]["carry"]
    assert res["label"] == "exploratory"
    assert r["hml_mean_monthly"] == pytest.approx(0.03)
    assert r["months"] == len(MONTHS) - 1
    assert r["known_premium"].startswith("Lustig")


def test_noisy_panel_is_significant_and_gate_reports():
    rx, sig, live = _toy_panel(0.02, seed=1)
    res = study.run_panel(rx, {"momentum": sig}, live, n_port=3, ic_monthly=0.06, n_eff=15)
    r = res["signals"]["momentum"]
    assert r["hml_t_nw"] > 5 and r["fm_t"] > 5
    assert r["median_live"] == 6
    assert r["power_gate"]["verdict"] == "blocked"  # 6 < 30 currencies


def test_runner_refuses_hold_out_rows_and_dead_currencies_are_masked():
    rx, sig, live = _toy_panel(0.0)
    late = pd.date_range("2016-01-31", periods=2, freq="ME")
    with pytest.raises(HoldoutLocked):
        study.run_panel(pd.concat([rx, rx.iloc[:2].set_axis(late)]), {"carry": sig}, live)
    u = Universe.load()
    m = u.mask(MONTHS, ["USD", "SIT", "SKK"])
    assert not m.loc["2008-06-30", "SIT"] and m.loc["2008-06-30", "SKK"]


def test_nw_tstat_lag_zero_is_plain_t():
    x = pd.Series(np.random.default_rng(3).normal(0.1, 1, 200))
    plain = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert study.nw_tstat(x, lag=0) == pytest.approx(plain)


# ---- French factor momentum -----------------------------------------------


def test_worked_example_factor_momentum():
    idx = pd.date_range("1990-01-31", periods=24, freq="ME")
    f = pd.DataFrame({"HML": [0.01] * 12 + [-0.01] * 12}, index=idx)
    sig = ef.factor_momentum_signal(f)
    assert sig["HML"].iloc[11:17].eq(1).all() and sig["HML"].iloc[17] == 0
    assert sig["HML"].iloc[18:].eq(-1).all()
    res = ef.factor_momentum(f)
    r = res["returns"]["HML"]
    assert r.loc[idx[12]:idx[17]].eq(-0.01).all()
    assert r.loc[idx[18]] == 0 and r.loc[idx[19]] == pytest.approx(0.01)


def test_factor_momentum_on_the_recorded_french_file_is_locked_until_registered():
    pytest.importorskip("qp_data")
    from qp_data import french

    raw = (REPO / "python" / "data" / "tests" / "fixtures" / "F-F_Research_Data_Factors.zip")
    f = french.parse_french(raw.read_bytes())[["Mkt-RF", "SMB", "HML"]]
    with pytest.raises(HoldoutLocked):
        ef.factor_momentum(f)
    res = ef.factor_momentum(HoldoutLock.training(f))
    assert res["label"] == "exploratory"
    assert res["returns"].index.max() < pd.Timestamp("2016-01-01")
    assert res["stats"]["EW"]["months"] > 1000
