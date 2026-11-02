#!/usr/bin/env python3
"""Generate the cross-language golden vectors for the equity VaR/ES engines.

This is the single source of the reference constants asserted by

* ``cpp/equity-var-engine/tests/test_cross_language.cpp`` (via the header
  ``tests/golden_vectors.hpp`` emitted by
  ``cpp/equity-var-engine/tools/gen_golden_header.py``), and
* ``rust/equity-var-engine/tests/test_cross_language.rs`` (via
  ``tests/golden/mod.rs`` emitted by ``rust/equity-var-engine/tools/gen_golden_rs.py``),
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
conv_*  the cross-family convention keys of spec S13 (same names in the FX
   JSON): S2 VaR/ES on the rounded sample P, Basel any window / alpha,
   Kupiec, Acerbi-Szekely Z2, Cornish-Fisher validity flags, Higham and clip
   repairs, EWMA seed window.

Usage::

    cd python/equity/03-var-es-engine
    PYTHONPATH=src python3 tests/golden/generate_golden.py [out.json]
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy
from eq_var.backtesting import (
    acerbi_szekely_z2,
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
)
from eq_var.expected_shortfall import expected_shortfall, normal_es, student_t_es
from eq_var.historical_var import (
    age_weighted_var,
    brw_weights,
    ewma_forecast,
    filtered_historical_var,
    historical_var,
)
from eq_var.monte_carlo_var import higham_nearest_correlation, repair_covariance
from eq_var.parametric_var import (
    cornish_fisher_valid,
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


def round2(x: float) -> float:
    """Spec S13 rounding, identical in every language: floor(x * 100 + 0.5) / 100."""
    return math.floor(x * 100.0 + 0.5) / 100.0


def conv_pnl() -> np.ndarray:
    """``P_i = round2(1000 sin(0.37 i + 0.1) (1 + 0.5 cos(0.11 i)))``, i = 0..249."""
    return np.array(
        [round2(1000.0 * math.sin(0.37 * i + 0.1) * (1.0 + 0.5 * math.cos(0.11 * i))) for i in range(250)]
    )


CONV_A = [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]
CONV_BASEL = [(250, 4, 0.01), (250, 5, 0.01), (250, 9, 0.01), (250, 10, 0.01), (500, 8, 0.01),
              (500, 9, 0.01), (500, 12, 0.01), (1000, 15, 0.01), (250, 12, 0.025)]
CONV_CF = {
    "conv_cf_valid_s0122_km0427": (0.122, -0.427, 0.01),
    "conv_cf_valid_s00_k00": (0.0, 0.0, 0.01),
    "conv_cf_valid_s10_k10": (1.0, 1.0, 0.01),
    "conv_cf_valid_sm0105_k80001": (-0.0105, 8.0001, 0.01),
    "conv_cf_valid_a001_s0122_km0427": (0.122, -0.427, 0.001),
}


def build_conv() -> dict[str, float]:
    """Spec S13 keys (prefix ``conv_``)."""
    out: dict[str, float] = {}
    p = conv_pnl()
    for a, tag in ((0.010, "a010"), (0.025, "a025"), (0.050, "a050")):
        out[f"conv_var_{tag}"] = historical_var(p, a)
        out[f"conv_es_{tag}"] = expected_shortfall(p, a)
    zone_code = {"green": 0.0, "yellow": 1.0, "red": 2.0}
    for n, x, a in CONV_BASEL:
        tag = f"n{n}_x{x}_a{round(1000 * a):03d}"
        r = basel_traffic_light(n_obs=n, n_exceptions=x, alpha=a)
        out[f"conv_basel_zone_{tag}"] = zone_code[r["zone"]]
        out[f"conv_basel_k_{tag}"] = r["multiplier"]
        out[f"conv_basel_cdf_{tag}"] = r["cumulative_prob"]
    for n, x in ((250, 5), (500, 9), (250, 0)):
        out[f"conv_kupiec_lr_n{n}_x{x}"] = kupiec_pof(n_obs=n, n_exceptions=x, alpha=0.01)["lr"]
    out["conv_as_z2"] = acerbi_szekely_z2(p, 1000.0, 1300.0, alpha=0.025, n_sims=1)["z2"]
    for key, (s, k, a) in CONV_CF.items():
        out[key] = 1.0 if cornish_fisher_valid(s, k, a) else 0.0
    h = higham_nearest_correlation(np.array(CONV_A))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = repair_covariance(np.array(CONV_A), "clip")
    for i, j in ((0, 1), (0, 2), (1, 2)):
        out[f"conv_higham_a{i}{j}"] = float(h[i, j])
        out[f"conv_clip_a{i}{j}"] = float(c[i, j])
    out["conv_ewma_sigma_last"] = ewma_forecast(p / 1e5, 0.94, seed_window=30)
    return out


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
    with warnings.catch_warnings():  # deprecated P&L-level FHS, kept one release
        warnings.simplefilter("ignore", DeprecationWarning)
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

    k7 = kupiec_pof(n_obs=250, n_exceptions=7, alpha=0.01)
    out["c_kupiec_lr_250_7"] = k7["lr"]
    out["c_kupiec_p_250_7"] = k7["pvalue"]
    k0 = kupiec_pof(n_obs=250, n_exceptions=0, alpha=0.01)
    out["c_kupiec_lr_250_0"] = k0["lr"]
    out["c_kupiec_p_250_0"] = k0["pvalue"]
    ex = case_c_exceptions()
    ind = christoffersen_independence(ex)
    out["c_christoffersen_ind_lr"] = ind["lr"]
    out["c_christoffersen_ind_p"] = ind["pvalue"]
    cc = christoffersen_cc(ex, 0.01)
    out["c_christoffersen_cc_lr"] = cc["lr"]
    out["c_christoffersen_cc_p"] = cc["pvalue"]

    out.update(build_conv())

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
            "conv": {
                "pnl": "P_i = round2(1000 sin(0.37 i + 0.1) (1 + 0.5 cos(0.11 i))), i = 0..249, "
                "round2(x) = floor(x * 100 + 0.5) / 100",
                "var_es": "spec S2 on P at alpha 0.010 / 0.025 / 0.050",
                "basel": "spec S4, (n, x, alpha) in the key; zone 0 green, 1 yellow, 2 red",
                "kupiec": "spec S3 LR at alpha 0.01",
                "as_z2": "spec S5 Z2 on P with VaR_t = 1000, ES_t = 1300, alpha 0.025",
                "cf_valid": "spec S6 flag (1 valid, 0 not); alpha 0.01 unless the key says a001",
                "cholesky": "spec S7 repairs of A = [[1,.9,.9],[.9,1,-.9],[.9,-.9,1]]; "
                "higham tolerance 1e-8 abs, clip 1e-12 rel",
                "ewma": "spec S8 one-step-ahead forecast sigma_n for the day after the sample, "
                "r_i = P_i / 1e5, lam 0.94, seed = mean(r^2) over the first 30",
            },
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__},
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
