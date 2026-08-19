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
broken and the C++/Rust golden headers must be regenerated deliberately.

Case A — book revaluation and plain/BRW historical VaR & ES on a
         deterministic sinusoidal factor history (no RNG).
Case B — closed-form parametric normal / Student-t VaR & ES from fixed
         exposures and covariance.
Case C — Kupiec / Christoffersen / Basel traffic-light statistics.
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
    """r[t][j] = s_j (sin(0.1 t + j) + 0.5 cos(0.05 t (j+1))) — no RNG."""
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
                           alpha=0.99, method="plain")
        assert r.var == pytest.approx(61919.80890587624, abs=1e-6)
        assert r.es == pytest.approx(62006.12006224847, abs=1e-6)

    def test_plain_historical_var_es_975(self, golden_book, golden_market,
                                         golden_returns) -> None:
        r = historical_var(golden_book, golden_market, golden_returns,
                           alpha=0.975, method="plain")
        assert r.var == pytest.approx(61237.42600889597, abs=1e-6)
        assert r.es == pytest.approx(61777.93608271857, abs=1e-6)

    def test_age_weighted_historical_var_es(self, golden_book, golden_market,
                                            golden_returns) -> None:
        r = historical_var(golden_book, golden_market, golden_returns,
                           alpha=0.99, method="age", decay=0.995)
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
        var, es = var_covar(exposures, cov, 0.99, 1.0, "normal")
        assert var == pytest.approx(153339.50441962917, rel=1e-12)
        assert es == pytest.approx(175675.6297200285, rel=1e-12)

    def test_student_t_df5_1d(self, exposures, cov) -> None:
        var, es = var_covar(exposures, cov, 0.99, 1.0, "t", 5.0)
        assert var == pytest.approx(171803.12389091405, rel=1e-12)
        assert es == pytest.approx(227327.5314974144, rel=1e-12)

    def test_normal_10d(self, exposures, cov) -> None:
        var, es = var_covar(exposures, cov, 0.99, 10.0, "normal")
        assert var == pytest.approx(484902.0892474838, rel=1e-12)
        assert es == pytest.approx(555535.1192996583, rel=1e-12)

    def test_ten_day_is_exactly_sqrt_ten_times_one_day(self, exposures,
                                                       cov) -> None:
        v1, e1 = var_covar(exposures, cov, 0.99, 1.0, "normal")
        v10, e10 = var_covar(exposures, cov, 0.99, 10.0, "normal")
        assert v10 == pytest.approx(v1 * np.sqrt(10.0), rel=1e-12)
        assert e10 == pytest.approx(e1 * np.sqrt(10.0), rel=1e-12)


class TestCaseCBacktestStatistics:
    def test_pattern_has_nine_exceptions(self, exceedances) -> None:
        assert int(exceedances.sum()) == 9

    def test_kupiec_pof(self) -> None:
        lr, p = kupiec_pof(8, 250, 0.99)
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
        tl = basel_traffic_light(n_exc, 250, 0.99)
        assert tl.cumulative_prob == pytest.approx(cum_prob, abs=1e-12)
        assert tl.zone == zone
        assert tl.multiplier == pytest.approx(mult)
