"""Golden vectors: the committed JSON must match live recomputation.

These vectors are the cross-language contract for the C++/Rust engines
(CONVENTIONS.md); this test guarantees the committed file never drifts
from the Python reference implementation.
"""

import json
import pathlib

import pytest

from fx_options import analytic_greeks

GOLDEN = pathlib.Path(__file__).parent / "golden" / "golden_vectors.json"

FIELD_MAP = {
    "price": "price", "delta_spot": "delta_spot",
    "delta_fwd": "delta_forward", "gamma": "gamma", "vega": "vega",
    "theta": "theta", "rho_d": "rho_domestic", "rho_f": "rho_foreign",
}


@pytest.fixture(scope="module")
def payload():
    return json.loads(GOLDEN.read_text())


def test_golden_file_shape(payload):
    assert payload["n_cases"] == len(payload["cases"]) == 30
    assert payload["tolerance"] == 1e-10


def test_golden_values_match_recomputation(payload):
    for case in payload["cases"]:
        inp = case["inputs"]
        g = analytic_greeks(inp["S"], inp["K"], inp["T"], inp["r_d"],
                            inp["r_f"], inp["sigma"], inp["type"]).as_dict()
        for json_key, attr in FIELD_MAP.items():
            assert case["outputs"][json_key] == pytest.approx(
                g[attr], abs=1e-10), (inp, json_key)


def test_golden_covers_required_regimes(payload):
    cases = [c["inputs"] for c in payload["cases"]]
    assert any(c["r_d"] < 0 and c["r_f"] < 0 for c in cases)  # negative rates
    assert any(c["sigma"] >= 0.30 for c in cases)             # EM vol
    assert any(c["T"] >= 5.0 for c in cases)                  # long-dated
    assert any(c["T"] < 0.05 for c in cases)                  # short-dated
    assert any(c["S"] > 100 for c in cases)                   # JPY-style level
    assert {c["type"] for c in cases} == {"call", "put"}
