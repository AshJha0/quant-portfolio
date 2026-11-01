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
conv_*  convention vectors shared by the equity and FX JSONs (spec S13):
   order-statistic VaR and Acerbi-Tasche ES on the sample
   ``P_i = round2(1000 sin(0.37 i + 0.1) (1 + 0.5 cos(0.11 i)))``,
   ``round2(x) = floor(100 x + 0.5) / 100``, i = 0..249; Basel zones and
   multipliers off the 250-day window; Kupiec LR; Acerbi-Szekely Z2;
   Cornish-Fisher validity flags; Higham and clip PSD repair; EWMA vol.

Tolerance
---------
The Python self-check compares the committed JSON with a fresh build to
``GOLDEN_ULPS`` units in the last place of ``max(|value|, scale)``.  Case A
values are P&L differences of book values of order 1e7 USD, so their
rounding error is set by the gross value, not by the difference: the
scale for ``a_*`` keys is 2e7.  ``conv_higham_*`` comes from an iterative
solver stopped at 1e-12 relative change, so it is compared at 1e-8
absolute.  Every other key is a short closed-form or order-statistic
computation, compared at ``GOLDEN_ULPS`` ulps of its own magnitude.

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

import platform
import warnings

import scipy

from fx_var.backtesting import (
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_independence,
    kupiec_pof,
)
from fx_var.book import Book, Forward, Market, Spot
from fx_var.expected_shortfall import empirical_var_es
from fx_var.historical_var import ewma_volatility, historical_var
from fx_var.monte_carlo_var import _repair
from fx_var.parametric_var import cornish_fisher_valid, var_covar

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

#: ulp bound for the Python self-check (see module docstring).
GOLDEN_ULPS = 16
#: magnitude floor for case A keys: gross book value scale in USD.
CASE_A_SCALE = 2e7

CONV_BASEL = [(250, 4, 0.01), (250, 5, 0.01), (250, 9, 0.01), (250, 10, 0.01),
              (500, 8, 0.01), (500, 9, 0.01), (500, 12, 0.01), (1000, 15, 0.01),
              (250, 12, 0.025)]
CONV_HIGHAM_A = [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]


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


def round2(x: float) -> float:
    """``floor(100 x + 0.5) / 100``: the same rounding in every language."""
    return float(np.floor(x * 100.0 + 0.5) / 100.0)


def conv_sample() -> np.ndarray:
    """Spec S13 sample ``P``: 250 deterministic P&L values."""
    return np.array([
        round2(1000.0 * np.sin(0.37 * i + 0.1) * (1.0 + 0.5 * np.cos(0.11 * i)))
        for i in range(250)
    ])


def build_conv() -> dict[str, float]:
    """Convention vectors (keys identical in the equity and FX JSONs)."""
    out: dict[str, float] = {}
    p = conv_sample()
    for a, tag in ((0.010, "a010"), (0.025, "a025"), (0.050, "a050")):
        out[f"conv_var_{tag}"], out[f"conv_es_{tag}"] = empirical_var_es(p, a)
    zone_code = {"green": 0.0, "yellow": 1.0, "red": 2.0}
    for n, x, a in CONV_BASEL:
        tl = basel_traffic_light(n_obs=n, n_exceptions=x, alpha=a)
        key = f"n{n}_x{x}_a{round(1000 * a):03d}"
        out[f"conv_basel_zone_{key}"] = zone_code[tl.zone]
        out[f"conv_basel_k_{key}"] = tl.multiplier
        out[f"conv_basel_cdf_{key}"] = tl.cumulative_prob
    for n, x in ((250, 5), (500, 9), (250, 0)):
        out[f"conv_kupiec_lr_n{n}_x{x}"] = kupiec_pof(n_obs=n, n_exceptions=x, alpha=0.01)[0]
    out["conv_as_z2"] = acerbi_szekely_z2(p, np.full(250, 1500.0), np.full(250, 1800.0),
                                          alpha=0.025, n_sims=1)["z2"]
    for (sk, ku, a), tag in (((0.122, -0.427, 0.01), "s0122_km0427"),
                             ((0.0, 0.0, 0.01), "s00_k00"),
                             ((1.0, 1.0, 0.01), "s10_k10"),
                             ((-0.0105, 8.0001, 0.01), "sm0105_k80001"),
                             ((0.122, -0.427, 0.001), "a001_s0122_km0427")):
        out[f"conv_cf_valid_{tag}"] = 1.0 if cornish_fisher_valid(sk, ku, a) else 0.0
    a3 = np.array(CONV_HIGHAM_A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hi = _repair(a3, "higham")
        cl = _repair(a3, "clip")
    for i, j in ((0, 1), (0, 2), (1, 2)):
        out[f"conv_higham_a{i}{j}"] = hi[i, j]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        out[f"conv_clip_a{i}{j}"] = cl[i, j]
    r = pd.DataFrame({"r": p / 1e5})
    _, nxt = ewma_volatility(r, 0.94, seed_window=30)
    out["conv_ewma_sigma_last"] = nxt["r"]
    return {k: float(v) for k, v in out.items()}


def build() -> dict[str, object]:
    """Compute every golden value.  Pure function of the fx_var package."""
    out: dict[str, object] = {}

    # ---- Case A: book revaluation + historical family ------------------- #
    market, book, rets = golden_market(), golden_book(), golden_returns()
    assert book.factors(market) == GOLDEN_FACTORS
    out["a_book_pnl_scenario_17"] = book.pnl(market, rets.iloc[17])
    r99 = historical_var(book, market, rets, alpha=0.01, method="plain")
    out["a_plain_var_99"], out["a_plain_es_99"] = r99.var, r99.es
    r975 = historical_var(book, market, rets, alpha=0.025, method="plain")
    out["a_plain_var_975"], out["a_plain_es_975"] = r975.var, r975.es
    rage = historical_var(book, market, rets, alpha=0.01, method="age", decay=0.995)
    out["a_age_var_99_decay_0995"], out["a_age_es_99_decay_0995"] = rage.var, rage.es

    # ---- Case B: parametric closed form --------------------------------- #
    w = golden_exposures()
    cov = golden_cov(w)
    out["b_normal_var_99_1d"], out["b_normal_es_99_1d"] = var_covar(w, cov, 0.01, 1.0, "normal")
    out["b_t5_var_99_1d"], out["b_t5_es_99_1d"] = var_covar(w, cov, 0.01, 1.0, "t", 5.0)
    out["b_normal_var_99_10d"], out["b_normal_es_99_10d"] = var_covar(
        w, cov, 0.01, 10.0, "normal"
    )

    # ---- Case C: backtest statistics ------------------------------------ #
    out["c_kupiec_lr_8_250"], out["c_kupiec_p_8_250"] = kupiec_pof(
        n_obs=250, n_exceptions=8, alpha=0.01
    )
    out["c_christoffersen_ind_lr"], out["c_christoffersen_ind_p"] = christoffersen_independence(
        golden_exceedances()
    )
    for x in (4, 5, 10):
        tl = basel_traffic_light(n_obs=250, n_exceptions=x, alpha=0.01)
        out[f"c_basel_cum_prob_{x}"] = tl.cumulative_prob
        out[f"c_basel_zone_{x}"] = tl.zone
        out[f"c_basel_multiplier_{x}"] = tl.multiplier

    values = {k: (v if isinstance(v, str) else float(v)) for k, v in out.items()}
    values.update(build_conv())
    return {
        "description": DESCRIPTION,
        "source": "python/fx/03-var-es-engine (fx_var)",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
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
                "kupiec": {"n_obs": 250, "n_exceptions": 8, "alpha": 0.01},
                "exceptions": "t % 37 == 5 plus days 100 and 101 (9 in 250)",
                "basel": {"n_obs": 250, "n_exceptions": [4, 5, 10], "alpha": 0.01},
            },
            "conv": {
                "sample": "P_i = round2(1000 sin(0.37 i + 0.1) (1 + 0.5 cos(0.11 i))), i=0..249, "
                          "round2(x) = floor(100 x + 0.5) / 100",
                "alphas": [0.010, 0.025, 0.050],
                "basel": [list(t) for t in CONV_BASEL],
                "kupiec": [[250, 5], [500, 9], [250, 0]],
                "as": {"var": 1500.0, "es": 1800.0, "alpha": 0.025},
                "higham_input": CONV_HIGHAM_A,
                "ewma": {"returns": "P_i / 1e5", "lambda": 0.94, "seed_window": 30},
            },
        },
        "tolerance": {
            "case_a_abs": 1e-6,
            "case_b_rel": 1e-8,
            "kupiec_lr_abs": 1e-10,
            "kupiec_p_abs": 1e-11,
            "christoffersen_abs": 1e-10,
            "basel_cum_prob_abs": 1e-12,
            "python_self_check_ulps": GOLDEN_ULPS,
            "conv_higham_abs": 1e-8,
            "conv_clip_rel": 1e-12,
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
