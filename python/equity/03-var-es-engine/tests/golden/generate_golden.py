#!/usr/bin/env python3
"""Generate the cross-language golden vectors for the equity VaR/ES engines.

This is the single source of the reference constants asserted by

* ``cpp/equity-var-engine/tests/test_cross_language.cpp`` (via the header
  ``tests/golden_vectors.hpp`` emitted by
  ``cpp/equity-var-engine/tools/gen_golden_header.py``), and
* ``rust/equity-var-engine/tests/test_cross_language.rs`` (via
  ``src/golden.rs`` emitted by ``rust/equity-var-engine/tools/gen_golden_rs.py``),
* ``tests/test_golden_vectors.py`` in this project (the Python-side lock).

All inputs are closed-form (sin/cos based) deterministic series, so every
language regenerates them independently and no RNG is involved.  Outputs
are written to ``golden_vectors.json`` next to this file with full
``repr()`` precision (17 significant digits round-trip to the identical
IEEE-754 double in C++ and Rust).

Cases
-----
A  historical family on ``pnl[t] = 100 sin(3t+1) + 0.5 t cos(t)``, t=0..99
B  parametric family on fixed exposures / vols / correlation
C  covariance estimators on ``r[t,j] = 0.01 sin(t+j) + 0.005 cos(2t-j)``
   (t=0..59, j=0..2) plus Kupiec / Christoffersen statistics.

Usage::

    cd python/equity/03-var-es-engine
    PYTHONPATH=src python3 tests/golden/generate_golden.py [out.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from eq_var.backtesting import christoffersen_cc, christoffersen_independence, kupiec_pof
from eq_var.expected_shortfall import expected_shortfall, normal_es, student_t_es
from eq_var.historical_var import (
    age_weighted_var,
    brw_weights,
    filtered_historical_var,
    historical_var,
)
from eq_var.parametric_var import (
    cornish_fisher_var,
    ewma_covariance,
    parametric_var,
    portfolio_sigma,
    sample_covariance,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "golden_vectors.json"

DESCRIPTION = (
    "Golden VaR/ES/backtest vectors for cross-validating the C++/Rust "
    "equity-var-engine against the eq_var Python reference. Inputs are "
    "closed-form sin/cos series (no RNG); tolerance for consumers: 1e-9 "
    "relative + 1e-12 absolute (libm-vs-numpy ulp noise in sin/cos)."
)


def case_a_pnl() -> np.ndarray:
    """``pnl[t] = 100 sin(3t + 1) + 0.5 t cos(t)``, t = 0..99."""
    t = np.arange(100, dtype=float)
    return 100.0 * np.sin(3.0 * t + 1.0) + 0.5 * t * np.cos(t)


def case_c_returns() -> np.ndarray:
    """``r[t, j] = 0.01 sin(t + j) + 0.005 cos(2t - j)``, t=0..59, j=0..2."""
    t = np.arange(60, dtype=float)[:, None]
    j = np.arange(3, dtype=float)[None, :]
    return 0.01 * np.sin(t + j) + 0.005 * np.cos(2.0 * t - j)


def case_c_exceptions() -> np.ndarray:
    """Exception pattern: ``t % 40 == 0`` plus days 100 and 101 (9 in 250)."""
    ex = np.zeros(250, dtype=bool)
    ex[::40] = True
    ex[100] = True
    ex[101] = True
    assert int(ex.sum()) == 9
    return ex


def covariance_from_vols(vols: np.ndarray, corr: np.ndarray) -> np.ndarray:
    v = np.asarray(vols, dtype=float)
    return np.outer(v, v) * np.asarray(corr, dtype=float)


def build() -> dict[str, object]:
    """Compute every golden value.  Pure function of the eq_var package."""
    out: dict[str, object] = {}

    # ---- Case A: historical family ------------------------------------- #
    pnl = case_a_pnl()
    out["a_hist_var_1pct"] = historical_var(pnl, 0.01)
    out["a_hist_var_5pct"] = historical_var(pnl, 0.05)
    out["a_es_5pct"] = expected_shortfall(pnl, 0.05)
    out["a_es_1pct"] = expected_shortfall(pnl, 0.01)
    out["a_brw_var_5pct_lam098"] = age_weighted_var(pnl, 0.05, 0.98)
    out["a_fhs_var_5pct_lam094"] = filtered_historical_var(pnl, 0.05, 0.94)
    out["a_brw_weights_5_lam098"] = [float(x) for x in brw_weights(5, 0.98)]

    # ---- Case B: parametric family ------------------------------------- #
    w = np.array([1.0e6, -5.0e5, 2.0e5])
    vols = np.array([0.01, 0.015, 0.02])
    corr = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.3], [0.25, 0.3, 1.0]])
    cov = covariance_from_vols(vols, corr)
    sigma = portfolio_sigma(w, cov)
    out["b_sigma"] = sigma
    out["b_var_normal_1pct"] = parametric_var(w, cov, 0.01)
    out["b_var_t6_1pct"] = parametric_var(w, cov, 0.01, dist="t", df=6.0)
    out["b_var_normal_1pct_10d"] = parametric_var(w, cov, 0.01, horizon_days=10)
    out["b_es_normal_2p5pct"] = normal_es(sigma, 0.025)
    out["b_es_t6_2p5pct"] = student_t_es(sigma, 0.025, 6.0)
    out["b_cf_var_1pct_skew_m05_kurt_1"] = cornish_fisher_var(sigma, 0.01, -0.5, 1.0)

    # ---- Case C: covariance estimators + backtest statistics ----------- #
    r = case_c_returns()
    s = sample_covariance(r)
    e = ewma_covariance(r, 0.94)
    out["c_sample_cov_00"] = float(s[0, 0])
    out["c_sample_cov_01"] = float(s[0, 1])
    out["c_sample_cov_12"] = float(s[1, 2])
    out["c_ewma_cov_00"] = float(e[0, 0])
    out["c_ewma_cov_01"] = float(e[0, 1])
    out["c_ewma_cov_22"] = float(e[2, 2])
    w_c = np.array([2.0e6, -1.0e6, 5.0e5])
    out["c_var_normal_1pct"] = parametric_var(w_c, s, 0.01)
    out["c_var_t8_5pct"] = parametric_var(w_c, s, 0.05, dist="t", df=8.0)

    k7 = kupiec_pof(250, 7, 0.01)
    out["c_kupiec_lr_250_7"] = k7["lr"]
    out["c_kupiec_p_250_7"] = k7["pvalue"]
    k0 = kupiec_pof(250, 0, 0.01)
    out["c_kupiec_lr_250_0"] = k0["lr"]
    out["c_kupiec_p_250_0"] = k0["pvalue"]
    ex = case_c_exceptions()
    ind = christoffersen_independence(ex)
    out["c_christoffersen_ind_lr"] = ind["lr"]
    out["c_christoffersen_ind_p"] = ind["pvalue"]
    cc = christoffersen_cc(ex, 0.01)
    out["c_christoffersen_cc_lr"] = cc["lr"]
    out["c_christoffersen_cc_p"] = cc["pvalue"]

    return {
        "description": DESCRIPTION,
        "source": "python/equity/03-var-es-engine (eq_var)",
        "inputs": {
            "case_a": "pnl[t] = 100 sin(3t + 1) + 0.5 t cos(t), t = 0..99",
            "case_b": {
                "exposures": [1.0e6, -5.0e5, 2.0e5],
                "vols": [0.01, 0.015, 0.02],
                "corr": [[1.0, 0.5, 0.25], [0.5, 1.0, 0.3], [0.25, 0.3, 1.0]],
            },
            "case_c": {
                "returns": "r[t, j] = 0.01 sin(t + j) + 0.005 cos(2t - j), t=0..59, j=0..2",
                "exposures": [2.0e6, -1.0e6, 5.0e5],
                "exceptions": "t % 40 == 0 plus days 100 and 101 (9 in 250)",
            },
        },
        "tolerance": {"rel": 1e-9, "abs": 1e-12},
        "values": {k: (float(v) if not isinstance(v, list) else v) for k, v in out.items()},
    }


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    data = build()
    out_path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {len(data['values'])} golden values to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
