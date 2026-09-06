#!/usr/bin/env python3
"""Generate the cross-language golden vectors for the FX VaR/ES engines.

This is the single source of the reference constants asserted by

* ``cpp/fx-var-engine/tests/test_golden_python.cpp`` (via the header
  ``tests/golden_vectors.hpp`` emitted by
  ``cpp/fx-var-engine/tools/gen_golden_header.py``),
* ``rust/fx-var-engine/tests/test_golden_python.rs`` (via
  ``tests/golden/mod.rs`` emitted by
  ``rust/fx-var-engine/tools/gen_golden_rs.py``), and
* ``tests/test_golden_vectors.py`` in this project (the Python-side lock).

All inputs are closed-form (sin/cos based) deterministic series, so every
language regenerates them independently and no RNG is involved.  Outputs
are written to ``golden_vectors.json`` next to this file with full
``repr()`` precision (17 significant digits round-trip to the identical
IEEE-754 double in C++ and Rust).

Cases
-----
A  book revaluation + plain / age-weighted historical VaR & ES on
   ``r[t][j] = s_j (sin(0.1 t + j) + 0.5 cos(0.05 t (j+1)))``, t=0..299,
   factors (FX:EUR, FX:JPY, IR:JPY, IR:USD), s = (0.006, 0.007, 4e-4, 5e-4)
B  closed-form parametric normal / Student-t VaR & ES from fixed exposures
   and covariance (``fx_var.parametric_var.var_covar``)
C  Kupiec / Christoffersen / Basel traffic-light statistics

Usage::

    cd python/fx/03-var-es-engine
    PYTHONPATH=src python3 tests/golden/generate_golden.py [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fx_var.backtesting import basel_traffic_light, christoffersen_independence, kupiec_pof
from fx_var.book import Book, Forward, Market, Spot
from fx_var.historical_var import historical_var
from fx_var.parametric_var import var_covar

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "golden_vectors.json"

DESCRIPTION = (
    "Golden VaR/ES/backtest vectors for cross-validating the C++/Rust "
    "fx-var-engine against the fx_var Python reference. Inputs are "
    "closed-form sin/cos series (no RNG). Tolerances for consumers: P&L-scale "
    "figures 1e-6 absolute (case A) / 1e-8 relative (case B); Kupiec LR 1e-10, "
    "p-values 1e-11 (Kupiec) and 1e-10 (Christoffersen); Basel cumulative "
    "probabilities 1e-12 absolute."
)

GOLDEN_FACTORS = ["FX:EUR", "FX:JPY", "IR:JPY", "IR:USD"]
GOLDEN_SCALES = [0.006, 0.007, 0.0004, 0.0005]
GOLDEN_N = 300


def golden_market() -> Market:
    return Market(
        {"EUR": 1.10, "JPY": 0.0090, "GBP": 1.27},
        {"USD": 0.050, "EUR": 0.030, "JPY": 0.001},
    )


def golden_book() -> Book:
    return Book(
        [
            Spot("EURUSD", 10_000_000.0),  # long 10m EUR at market
            Forward("USDJPY", 5_000_000.0, 0.5),  # ATM CIP 6m forward
            Spot("EURJPY", -3_000_000.0),  # short 3m EUR cross
        ]
    )


def golden_returns() -> pd.DataFrame:
    """``r[t][j] = s_j (sin(0.1 t + j) + 0.5 cos(0.05 t (j+1)))`` -- no RNG."""
    data = np.array(
        [
            [
                GOLDEN_SCALES[j] * (np.sin(0.1 * t + j) + 0.5 * np.cos(0.05 * t * (j + 1)))
                for j in range(4)
            ]
            for t in range(GOLDEN_N)
        ]
    )
    return pd.DataFrame(data, columns=GOLDEN_FACTORS)


def golden_exposures() -> pd.Series:
    return pd.Series({"FX:EUR": 11e6, "FX:JPY": -4.5e6, "IR:USD": -2.4e6})


def golden_cov(exposures: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            [3.6e-5, 1.1e-5, -2.0e-6],
            [1.1e-5, 4.9e-5, -1.0e-6],
            [-2.0e-6, -1.0e-6, 2.5e-7],
        ],
        index=exposures.index,
        columns=exposures.index,
    )


def golden_exceedances() -> np.ndarray:
    """Exception pattern: ``t % 37 == 5`` plus days 100 and 101 (9 in 250)."""
    e = np.zeros(250, dtype=int)
    e[[t for t in range(250) if t % 37 == 5]] = 1
    e[100] = 1
    e[101] = 1
    assert int(e.sum()) == 9
    return e


def build() -> dict[str, object]:
    """Compute every golden value.  Pure function of the fx_var package."""
    out: dict[str, object] = {}

    # ---- Case A: book revaluation + historical family ------------------- #
    market, book, rets = golden_market(), golden_book(), golden_returns()
    assert book.factors(market) == GOLDEN_FACTORS
    out["a_book_pnl_scenario_17"] = book.pnl(market, rets.iloc[17])
    r99 = historical_var(book, market, rets, alpha=0.99, method="plain")
    out["a_plain_var_99"], out["a_plain_es_99"] = r99.var, r99.es
    r975 = historical_var(book, market, rets, alpha=0.975, method="plain")
    out["a_plain_var_975"], out["a_plain_es_975"] = r975.var, r975.es
    rage = historical_var(book, market, rets, alpha=0.99, method="age", decay=0.995)
    out["a_age_var_99_decay_0995"], out["a_age_es_99_decay_0995"] = rage.var, rage.es

    # ---- Case B: parametric closed form --------------------------------- #
    w = golden_exposures()
    cov = golden_cov(w)
    out["b_normal_var_99_1d"], out["b_normal_es_99_1d"] = var_covar(w, cov, 0.99, 1.0, "normal")
    out["b_t5_var_99_1d"], out["b_t5_es_99_1d"] = var_covar(w, cov, 0.99, 1.0, "t", 5.0)
    out["b_normal_var_99_10d"], out["b_normal_es_99_10d"] = var_covar(
        w, cov, 0.99, 10.0, "normal"
    )

    # ---- Case C: backtest statistics ------------------------------------ #
    out["c_kupiec_lr_8_250"], out["c_kupiec_p_8_250"] = kupiec_pof(8, 250, 0.99)
    out["c_christoffersen_ind_lr"], out["c_christoffersen_ind_p"] = christoffersen_independence(
        golden_exceedances()
    )
    for x in (4, 5, 10):
        tl = basel_traffic_light(x, 250, 0.99)
        out[f"c_basel_cum_prob_{x}"] = tl.cumulative_prob
        out[f"c_basel_zone_{x}"] = tl.zone
        out[f"c_basel_multiplier_{x}"] = tl.multiplier

    values = {k: (v if isinstance(v, str) else float(v)) for k, v in out.items()}
    return {
        "description": DESCRIPTION,
        "source": "python/fx/03-var-es-engine (fx_var)",
        "inputs": {
            "case_a": {
                "market": {
                    "spots": {"EUR": 1.10, "JPY": 0.0090, "GBP": 1.27},
                    "rates": {"USD": 0.050, "EUR": 0.030, "JPY": 0.001},
                },
                "book": [
                    ["Spot", "EURUSD", 10_000_000.0],
                    ["Forward", "USDJPY", 5_000_000.0, 0.5],
                    ["Spot", "EURJPY", -3_000_000.0],
                ],
                "factors": GOLDEN_FACTORS,
                "returns": "r[t][j] = s_j (sin(0.1 t + j) + 0.5 cos(0.05 t (j+1))), t=0..299",
                "scales": GOLDEN_SCALES,
            },
            "case_b": {
                "exposures": {k: float(v) for k, v in w.items()},
                "cov": cov.values.tolist(),
            },
            "case_c": {
                "kupiec": [8, 250, 0.99],
                "exceptions": "t % 37 == 5 plus days 100 and 101 (9 in 250)",
                "basel": [4, 5, 10],
            },
        },
        "tolerance": {
            "case_a_abs": 1e-6,
            "case_b_rel": 1e-8,
            "kupiec_lr_abs": 1e-10,
            "kupiec_p_abs": 1e-11,
            "christoffersen_abs": 1e-10,
            "basel_cum_prob_abs": 1e-12,
        },
        "values": values,
    }


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    data = build()
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {len(data['values'])} golden values to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
