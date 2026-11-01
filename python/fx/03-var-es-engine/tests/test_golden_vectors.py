"""Python-side lock on the cross-language golden vectors.

The C++ (`cpp/fx-var-engine/tests/test_golden_python.cpp`) and Rust
(`rust/fx-var-engine/tests/test_golden_python.rs`) engines assert hard-coded
constants that were generated from *this* package.  Until now nothing on the
Python side pinned those numbers, so a refactor here could silently move the
reference and the only symptom would be a C++/Rust test failing in a
different repository directory.

This module reproduces the three golden cases exactly as the C++ file
documents them and asserts the same constants to the same tolerances.  If a
change to `fx_var` moves any number below, the cross-language contract is
broken and the C++/Rust golden headers must be regenerated on purpose.

Case A: book revaluation and plain/BRW historical VaR & ES on a
         deterministic sinusoidal factor history (no RNG).
Case B: closed-form parametric normal / Student-t VaR & ES from fixed
         exposures and covariance.
Case C: Kupiec / Christoffersen / Basel traffic-light statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fx_var.backtesting import (
    basel_traffic_light,
    christoffersen_independence,
    kupiec_pof,
)
from fx_var.book import Book, Forward, Market, Spot
from fx_var.historical_var import historical_var
from fx_var.parametric_var import var_covar

GOLDEN_FACTORS = ["FX:EUR", "FX:JPY", "IR:JPY", "IR:USD"]
GOLDEN_SCALES = [0.006, 0.007, 0.0004, 0.0005]
GOLDEN_N = 300


@pytest.fixture(scope="module")
def golden_market() -> Market:
    return Market({"EUR": 1.10, "JPY": 0.0090, "GBP": 1.27},
                  {"USD": 0.050, "EUR": 0.030, "JPY": 0.001})


@pytest.fixture(scope="module")
def golden_book() -> Book:
    return Book([
        Spot("EURUSD", 10_000_000.0),          # long 10m EUR at market
        Forward("USDJPY", 5_000_000.0, 0.5),   # ATM CIP 6m forward
        Spot("EURJPY", -3_000_000.0),          # short 3m EUR cross
    ])


@pytest.fixture(scope="module")
def golden_returns() -> pd.DataFrame:
    """r[t][j] = s_j (sin(0.1 t + j) + 0.5 cos(0.05 t (j+1))): no RNG."""
    data = np.array([
        [GOLDEN_SCALES[j] * (np.sin(0.1 * t + j)
                             + 0.5 * np.cos(0.05 * t * (j + 1)))
         for j in range(4)]
        for t in range(GOLDEN_N)
    ])
    return pd.DataFrame(data, columns=GOLDEN_FACTORS)


class TestCaseABookAndHistorical:
    def test_factor_enumeration_order_is_the_golden_order(
            self, golden_book, golden_market) -> None:
        # The C++/Rust engines index the returns matrix positionally, so the
        # factor order is itself part of the contract.
        assert golden_book.factors(golden_market) == GOLDEN_FACTORS

    def test_single_scenario_pnl(self, golden_book, golden_market,
                                 golden_returns) -> None:
        got = golden_book.pnl(golden_market, golden_returns.iloc[17])
        assert got == pytest.approx(58177.37489810074, abs=1e-6)

    def test_plain_historical_var_es_99(self, golden_book, golden_market,
                                        golden_returns) -> None:
        r = historical_var(golden_book, golden_market, golden_returns,
                           alpha=0.01, method="plain")
        assert r.var == pytest.approx(61919.80890587624, abs=1e-6)
        assert r.es == pytest.approx(62006.12006224847, abs=1e-6)

    def test_plain_historical_var_es_975(self, golden_book, golden_market,
                                         golden_returns) -> None:
        r = historical_var(golden_book, golden_market, golden_returns,
                           alpha=0.025, method="plain")
        assert r.var == pytest.approx(61237.42600889597, abs=1e-6)
        assert r.es == pytest.approx(61777.93608271857, abs=1e-6)

    def test_age_weighted_historical_var_es(self, golden_book, golden_market,
                                            golden_returns) -> None:
        r = historical_var(golden_book, golden_market, golden_returns,
                           alpha=0.01, method="age", decay=0.995)
        assert r.var == pytest.approx(61874.26268531149, abs=1e-6)
        assert r.es == pytest.approx(61977.52496594109, abs=1e-6)


@pytest.fixture(scope="module")
def exposures() -> pd.Series:
    return pd.Series({"FX:EUR": 11e6, "FX:JPY": -4.5e6, "IR:USD": -2.4e6})


@pytest.fixture(scope="module")
def cov(exposures) -> pd.DataFrame:
    return pd.DataFrame(
        [[3.6e-5, 1.1e-5, -2.0e-6],
         [1.1e-5, 4.9e-5, -1.0e-6],
         [-2.0e-6, -1.0e-6, 2.5e-7]],
        index=exposures.index, columns=exposures.index)


@pytest.fixture(scope="module")
def exceedances() -> np.ndarray:
    """Deterministic exception pattern: t % 37 == 5 plus days 100 and 101."""
    e = np.zeros(250, dtype=int)
    e[[t for t in range(250) if t % 37 == 5]] = 1
    e[100] = 1
    e[101] = 1
    return e


class TestCaseBParametricClosedForm:
    def test_normal_1d(self, exposures, cov) -> None:
        var, es = var_covar(exposures, cov, 0.01, 1.0, "normal")
        assert var == pytest.approx(153339.50441962917, rel=1e-12)
        assert es == pytest.approx(175675.6297200285, rel=1e-12)

    def test_student_t_df5_1d(self, exposures, cov) -> None:
        var, es = var_covar(exposures, cov, 0.01, 1.0, "t", 5.0)
        assert var == pytest.approx(171803.12389091405, rel=1e-12)
        assert es == pytest.approx(227327.5314974144, rel=1e-12)

    def test_normal_10d(self, exposures, cov) -> None:
        var, es = var_covar(exposures, cov, 0.01, 10.0, "normal")
        assert var == pytest.approx(484902.0892474838, rel=1e-12)
        assert es == pytest.approx(555535.1192996583, rel=1e-12)

    def test_ten_day_is_exactly_sqrt_ten_times_one_day(self, exposures,
                                                       cov) -> None:
        v1, e1 = var_covar(exposures, cov, 0.01, 1.0, "normal")
        v10, e10 = var_covar(exposures, cov, 0.01, 10.0, "normal")
        assert v10 == pytest.approx(v1 * np.sqrt(10.0), rel=1e-12)
        assert e10 == pytest.approx(e1 * np.sqrt(10.0), rel=1e-12)


class TestCaseCBacktestStatistics:
    def test_pattern_has_nine_exceptions(self, exceedances) -> None:
        assert int(exceedances.sum()) == 9

    def test_kupiec_pof(self) -> None:
        lr, p = kupiec_pof(n_obs=250, n_exceptions=8, alpha=0.01)
        assert lr == pytest.approx(7.7335507244945205, abs=1e-10)
        assert p == pytest.approx(0.0054204051941277994, abs=1e-11)

    def test_christoffersen_independence(self, exceedances) -> None:
        lr, p = christoffersen_independence(exceedances)
        assert lr == pytest.approx(1.0063610339314124, abs=1e-10)
        assert p == pytest.approx(0.3157762037622499, abs=1e-10)

    @pytest.mark.parametrize(
        "n_exc, cum_prob, zone, mult",
        [
            (4, 0.8921876269036249, "green", 3.00),
            (5, 0.9588168159301514, "yellow", 3.40),
            (10, 0.999946101370953, "red", 4.00),
        ],
    )
    def test_basel_traffic_light(self, n_exc, cum_prob, zone, mult) -> None:
        tl = basel_traffic_light(n_obs=250, n_exceptions=n_exc, alpha=0.01)
        assert tl.cumulative_prob == pytest.approx(cum_prob, abs=1e-12)
        assert tl.zone == zone
        assert tl.multiplier == pytest.approx(mult)


# ---------------------------------------------------------------------------
# The committed ``tests/golden/golden_vectors.json`` (consumed by
# cpp/fx-var-engine/tools/gen_golden_header.py and
# rust/fx-var-engine/tools/gen_golden_rs.py) must match what the generator
# produces from the live package, to the ulp bound documented in the
# generator (not bit identity: libm and numpy versions move the last bits).
# ---------------------------------------------------------------------------
import importlib.util
import json
from pathlib import Path

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "fx_var_generate_golden", _GOLDEN_DIR / "generate_golden.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads((_GOLDEN_DIR / "golden_vectors.json").read_text())


@pytest.fixture(scope="module")
def regenerated(generator) -> dict:
    return generator.build()


def _within_ulps(key: str, a: float, b: float, gen) -> bool:
    """The generator's documented tolerance for ``key``."""
    if key.startswith("conv_higham_"):
        return abs(a - b) <= 1e-8
    scale = gen.CASE_A_SCALE if key.startswith("a_") else 0.0
    mag = max(abs(a), abs(b), scale)
    return abs(a - b) <= gen.GOLDEN_ULPS * float(np.spacing(mag))


def _minor(v: str) -> tuple:
    return tuple(v.split(".")[:2])


def _same_minor_versions(recorded: dict) -> bool:
    import scipy

    return (_minor(recorded["numpy"]) == _minor(np.__version__)
            and _minor(recorded["scipy"]) == _minor(scipy.__version__))


class TestCommittedJsonIsReproducible:
    def test_same_keys(self, committed, regenerated):
        assert list(committed["values"]) == list(regenerated["values"])

    def test_every_value_within_documented_ulp_bound(self, committed, regenerated,
                                                     generator):
        # The ulp bound holds against the numpy/scipy minor versions recorded
        # in the JSON. Other versions move special functions by up to ~1e2
        # ulps, so there the check uses a 1e-10 relative bound instead.
        same = _same_minor_versions(committed["versions"])
        for k, v in regenerated["values"].items():
            c = committed["values"][k]
            if isinstance(v, str):
                assert c == v, k
            elif same:
                assert _within_ulps(k, c, v, generator), f"{k}: {c!r} vs {v!r}"
            else:
                tol = 1e-8 if k.startswith("conv_higham_") else 1e-10 * max(abs(c), abs(v), 1e-300)
                assert abs(c - v) <= tol, f"{k}: {c!r} vs {v!r}"

    def test_ulp_bound_is_not_vacuous(self, generator):
        # 16 ulps of 2e7 is 5.96e-08: a 1e-6 move in a case A value fails.
        assert not _within_ulps("a_plain_var_99", 61919.0, 61919.0 + 1e-6, generator)
        assert not _within_ulps("b_normal_var_99_1d", 1.0, 1.0 + 1e-14, generator)

    def test_versions_recorded(self, committed):
        assert set(committed["versions"]) == {"python", "numpy", "scipy"}

    def test_json_matches_the_constants_pinned_above(self, committed, generator):
        v = committed["values"]
        pinned = {
            "a_book_pnl_scenario_17": 58177.37489810074,
            "a_plain_var_99": 61919.80890587624,
            "a_age_es_99_decay_0995": 61977.52496594109,
            "b_t5_es_99_1d": 227327.5314974144,
            "b_normal_var_99_10d": 484902.0892474838,
            "c_kupiec_lr_8_250": 7.7335507244945205,
            "c_christoffersen_ind_p": 0.3157762037622499,
            "c_basel_cum_prob_10": 0.999946101370953,
        }
        for k, want in pinned.items():
            assert _within_ulps(k, v[k], want, generator), (k, v[k], want)
        assert (v["c_basel_zone_4"], v["c_basel_zone_5"], v["c_basel_zone_10"]) == (
            "green", "yellow", "red")


@pytest.fixture(scope="module")
def p(generator):
    return generator.conv_sample()


class TestConventionVectors:
    """conv_* values checked against independent hand computations."""

    def test_sample_rounding_rule(self, p):
        i = np.arange(250)
        raw = 1000.0 * np.sin(0.37 * i + 0.1) * (1.0 + 0.5 * np.cos(0.11 * i))
        np.testing.assert_array_equal(p, np.floor(raw * 100.0 + 0.5) / 100.0)

    @pytest.mark.parametrize("alpha, tag, m", [(0.010, "a010", 3), (0.025, "a025", 7),
                                                (0.050, "a050", 13)])
    def test_var_es_by_hand(self, committed, p, alpha, tag, m):
        losses = np.sort(-p)[::-1]
        k = 250 * alpha
        assert m == int(np.ceil(k))
        var = losses[m - 1]
        es = (losses[: m - 1].sum() + (k - (m - 1)) * losses[m - 1]) / k
        v = committed["values"]
        assert v[f"conv_var_{tag}"] == var
        assert v[f"conv_es_{tag}"] == pytest.approx(es, rel=4 * np.finfo(float).eps)

    @pytest.mark.parametrize("n, x, a, zone, mult", [
        (250, 4, 0.01, 0, 3.0), (250, 5, 0.01, 1, 3.4), (250, 9, 0.01, 1, 3.85),
        (250, 10, 0.01, 2, 4.0), (500, 8, 0.01, 0, 3.0), (500, 9, 0.01, 1, 3.5),
        (500, 12, 0.01, 1, 3.75), (1000, 15, 0.01, 1, 3.4), (250, 12, 0.025, 1, 3.65),
    ])
    def test_basel_against_scipy_binomial(self, committed, n, x, a, zone, mult):
        from scipy.stats import binom

        key = f"n{n}_x{x}_a{round(1000 * a):03d}"
        v = committed["values"]
        assert v[f"conv_basel_cdf_{key}"] == pytest.approx(binom.cdf(x, n, a), abs=1e-12)
        assert v[f"conv_basel_zone_{key}"] == zone
        assert v[f"conv_basel_k_{key}"] == pytest.approx(mult, abs=1e-15)

    @pytest.mark.parametrize("n, x", [(250, 5), (500, 9), (250, 0)])
    def test_kupiec_by_hand(self, committed, n, x):
        a, ph = 0.01, x / n
        ll0 = (n - x) * np.log(1 - a) + (x * np.log(a) if x else 0.0)
        ll1 = ((n - x) * np.log(1 - ph) if x < n else 0.0) + (x * np.log(ph) if x else 0.0)
        want = -2.0 * (ll0 - ll1)
        assert committed["values"][f"conv_kupiec_lr_n{n}_x{x}"] == pytest.approx(want, rel=1e-13)

    def test_as_z2_by_hand(self, committed, p):
        hit = p < -1000.0
        want = float(np.sum(p * hit / 1300.0)) / (250 * 0.025) + 1.0
        assert committed["values"]["conv_as_z2"] == pytest.approx(want, abs=1e-15)

    def test_ewma_by_hand(self, committed, p):
        r = p / 1e5
        s2 = float(np.mean(r[:30] ** 2))
        for x in r:
            s2 = 0.94 * s2 + 0.06 * x * x
        assert committed["values"]["conv_ewma_sigma_last"] == pytest.approx(np.sqrt(s2), rel=1e-14)

    def test_cf_flags(self, committed):
        v = committed["values"]
        assert v["conv_cf_valid_s0122_km0427"] == 1.0
        assert v["conv_cf_valid_a001_s0122_km0427"] == 1.0
        assert v["conv_cf_valid_s00_k00"] == 1.0
        assert v["conv_cf_valid_s10_k10"] == 1.0
        assert v["conv_cf_valid_sm0105_k80001"] == 0.0

    def test_repaired_matrices_are_valid_correlations(self, committed):
        v = committed["values"]
        for kind in ("higham", "clip"):
            c = np.eye(3)
            c[0, 1] = c[1, 0] = v[f"conv_{kind}_a01"]
            c[0, 2] = c[2, 0] = v[f"conv_{kind}_a02"]
            c[1, 2] = c[2, 1] = v[f"conv_{kind}_a12"]
            assert np.linalg.eigvalsh(c)[0] > -1e-10
        # The input is far from PSD (smallest eigenvalue -0.8).
        assert np.linalg.eigvalsh(np.array([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]]))[0] < -0.79
