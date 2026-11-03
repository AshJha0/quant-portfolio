"""W14c historical scenario set: shocks reproduce the source series on their windows."""

from __future__ import annotations

import csv
import dataclasses
import math
import os
from pathlib import Path

import pytest
from qp_core.scenarios import (
    DESCRIPTIONS,
    SERIES,
    apply_to_quotes,
    apply_to_snapshot,
    reverse_stress_scale,
    scenario,
    scenario_names,
)

EXPECTED = ("gfc_2008", "eur_2011", "chf_2015", "gbp_2016", "covid_2020", "gilts_2022")

# Hand-copied from the raw source files on 2026-11-03 (independent of the
# generated module): (scenario, series, start value, end value).
SPOT_CHECKS = [
    ("chf_2015", "EURCHF", 1.2010, 1.0280),
    ("gilts_2022", "GBP_ZERO_10Y", 0.035516, 0.044341),
]


def test_the_six_planned_episodes_exist_with_descriptions():
    assert scenario_names() == EXPECTED
    assert set(DESCRIPTIONS) == set(EXPECTED)
    with pytest.raises(KeyError, match="choose from"):
        scenario("dotcom_2000")


@pytest.mark.parametrize("name", EXPECTED)
def test_shock_moves_start_quotes_onto_end_quotes(name):
    sc = scenario(name)
    start = {s: obs[0][1] for s, obs in sc.observations.items()}
    end = {s: obs[-1][1] for s, obs in sc.observations.items()}
    moved = apply_to_quotes(start, sc.shocks)
    for s in start:
        # x0 + (x1 - x0) and x0 * exp(ln(x1/x0)) are exact up to rounding.
        assert moved[s] == pytest.approx(end[s], rel=1e-14, abs=1e-15), s
        assert sc.observations[s][0][0] == sc.start
        assert sc.observations[s][-1][0] == sc.end


@pytest.mark.parametrize("name", EXPECTED)
def test_path_shocks_reproduce_every_observation(name):
    sc = scenario(name)
    for s, path in sc.path_shocks().items():
        x0 = sc.observations[s][0][1]
        for (d, dx), (d_obs, x) in zip(path, sc.observations[s], strict=True):
            assert d == d_obs
            rebuilt = x0 + dx if SERIES[s].kind == "absolute" else x0 * math.exp(dx)
            assert rebuilt == pytest.approx(x, rel=1e-14, abs=1e-15)
        dates = [d for d, _ in path]
        assert dates == sorted(dates) and len(set(dates)) == len(dates)


@pytest.mark.parametrize(("name", "series", "x0", "x1"), SPOT_CHECKS)
def test_hand_copied_source_values(name, series, x0, x1):
    sc = scenario(name)
    assert sc.observations[series][0][1] == x0
    assert sc.observations[series][-1][1] == x1


def test_worked_example_chf_2015():
    dx = scenario("chf_2015").shocks["EURCHF"]
    assert dx == pytest.approx(math.log(1.0280 / 1.2010))
    assert round(dx, 4) == -0.1555
    assert apply_to_quotes({"EURCHF": 1.2010}, {"EURCHF": dx})["EURCHF"] == pytest.approx(1.0280)


def test_apply_with_mapping_and_missing_quote():
    sc = scenario("gbp_2016")
    quotes = {"EUR/GBP": 0.76, "EUR/USD": 1.13, "VIX": 17.0, "other": 5.0}
    out = apply_to_quotes(quotes, sc.shocks, {"EURGBP": "EUR/GBP", "EURUSD": "EUR/USD"})
    assert out["other"] == 5.0
    assert out["EUR/GBP"] > quotes["EUR/GBP"]  # sterling fell
    with pytest.raises(KeyError, match="no 'EURGBP'"):
        apply_to_quotes({"VIX": 1.0}, sc.shocks)


def test_apply_to_snapshot_seam():
    @dataclasses.dataclass(frozen=True)
    class Snap:
        asof: str
        quotes: dict

    snap = Snap("2020-02-21", {"VIX": 17.08, "EURUSD": 1.0848, "USD_ZERO_2Y": 0.013,
                               "USD_ZERO_10Y": 0.015})
    sc = scenario("covid_2020")
    out = apply_to_snapshot(snap, sc.shocks)
    assert out.asof == snap.asof and out is not snap
    assert out.quotes["VIX"] == pytest.approx(17.08 + sc.shocks["VIX"])
    with pytest.raises(TypeError):
        apply_to_snapshot({"quotes": {}}, sc.shocks)


def test_reverse_stress_scale_on_a_linear_book():
    sc = scenario("gilts_2022")
    dv01 = {"GBP_ZERO_10Y": 1_000_000.0, "GBP_ZERO_20Y": 2_000_000.0, "EURGBP": 0.0}

    def loss(shocks):
        return sum(dv01[k] * v for k, v in shocks.items())

    full = loss(sc.shocks)
    k = reverse_stress_scale(sc, loss, 0.5 * full)
    assert k == pytest.approx(0.5, abs=1e-9)
    assert reverse_stress_scale(sc, loss, -1.0) == 0.0
    with pytest.raises(ValueError, match="stays below"):
        reverse_stress_scale(sc, loss, 100 * full)


_SRC = os.environ.get("QP_SCENARIO_SOURCE_DIR")


@pytest.mark.skipif(not _SRC, reason="set QP_SCENARIO_SOURCE_DIR to the raw downloads to re-verify")
def test_against_raw_ecb_download():  # pragma: no cover - optional, network-free re-check
    rows = {}
    with open(Path(_SRC) / "fx.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["OBS_VALUE"]:
                rows[(r["CURRENCY"], r["TIME_PERIOD"])] = float(r["OBS_VALUE"])
    for name in EXPECTED:
        for s, obs in scenario(name).observations.items():
            if s.startswith("EUR"):
                for d, v in obs:
                    assert rows[(s[3:], d)] == v
