"""Python-side lock on the cross-language golden vectors (TEST-3).

``cpp/equity-var-engine/tests/test_cross_language.cpp`` and
``rust/equity-var-engine/tests/test_cross_language.rs`` assert constants
generated from *this* package by ``tests/golden/generate_golden.py`` and
committed to ``tests/golden/golden_vectors.json``.  Until now nothing on
the Python side pinned those numbers, so a refactor of ``eq_var`` could
silently move the reference and the only symptom would be a C++/Rust build
failing in a different directory.

Two things are checked here:

1. the committed JSON is exactly what the generator produces from the
   current package (``build()`` is a pure function of ``eq_var``); and
2. the reference values are what the C++/Rust sources hard-code, to the
   tolerance both engines use (1e-9 relative + 1e-12 absolute).

If a deliberate change to ``eq_var`` moves a number, regenerate with::

    PYTHONPATH=src python3 tests/golden/generate_golden.py
    python3 ../../../cpp/equity-var-engine/tools/gen_golden_header.py
    python3 ../../../rust/equity-var-engine/tools/gen_golden_rs.py

and commit all three outputs together.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
GOLDEN_JSON = GOLDEN_DIR / "golden_vectors.json"

REL, ABS = 1e-9, 1e-12


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "eq_var_generate_golden", GOLDEN_DIR / "generate_golden.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(GOLDEN_JSON.read_text())


@pytest.fixture(scope="module")
def regenerated() -> dict:
    return _load_generator().build()


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL, abs_tol=ABS)


class TestCommittedJsonIsReproducible:
    def test_same_keys(self, committed, regenerated):
        assert list(committed["values"]) == list(regenerated["values"])

    def test_every_value_bit_identical(self, committed, regenerated):
        # ``repr()`` round-trips doubles exactly, so the committed file must
        # match the live package to the bit, not merely to tolerance.
        for k, v in regenerated["values"].items():
            c = committed["values"][k]
            if isinstance(v, list):
                assert c == v, k
            else:
                assert c == v, f"{k}: committed {c!r} != regenerated {v!r}"

    def test_tolerance_is_the_one_the_engines_use(self, committed):
        assert committed["tolerance"] == {"rel": REL, "abs": ABS}

    def test_inputs_documented(self, committed):
        assert "sin(3t + 1)" in committed["inputs"]["case_a"]
        assert committed["inputs"]["case_b"]["exposures"] == [1.0e6, -5.0e5, 2.0e5]


# The constants exactly as they appear in test_cross_language.cpp / .rs.
CPP_RUST_CONSTANTS = {
    "a_hist_var_1pct": 1.224129222375264e02,
    "a_hist_var_5pct": 1.045522835374927e02,
    "a_es_5pct": 1.226373207703405e02,
    "a_es_1pct": 1.417568107549531e02,
    "a_brw_var_5pct_lam098": 1.082348601407293e02,
    "a_fhs_var_5pct_lam094": 1.093777910164513e02,
    "b_sigma": 9.962429422585637e03,
    "b_var_normal_1pct": 2.317607650751402e04,
    "b_var_t6_1pct": 2.556337478743866e04,
    "b_var_normal_1pct_10d": 7.328918899006478e04,
    "b_es_normal_2p5pct": 2.329019532123021e04,
    "b_es_t6_2p5pct": 2.648647588170394e04,
    "b_cf_var_1pct_skew_m05_kurt_1": 2.823062626871169e04,
    "c_sample_cov_00": 6.197010057346647e-05,
    "c_sample_cov_01": 3.367477490492125e-05,
    "c_sample_cov_12": 3.643904950084309e-05,
    "c_ewma_cov_00": 6.085653794066718e-05,
    "c_ewma_cov_01": 3.159039791408869e-05,
    "c_ewma_cov_22": 6.695548359317170e-05,
    "c_var_normal_1pct": 2.403056180968714e04,
    "c_var_t8_5pct": 1.663517215790393e04,
    "c_kupiec_lr_250_7": 5.496990447792683e00,
    "c_kupiec_p_250_7": 1.904923089052653e-02,
    "c_kupiec_lr_250_0": 5.025167926750726e00,
    "c_kupiec_p_250_0": 2.498150305344973e-02,
    "c_christoffersen_ind_lr": 1.189356349111719e00,
    "c_christoffersen_ind_p": 2.754594267438438e-01,
    "c_christoffersen_cc_lr": 1.141838698170947e01,
    "c_christoffersen_cc_p": 3.315345323267836e-03,
}
CPP_RUST_BRW_WEIGHTS = [
    1.920016255921656e-01,
    1.959200261144547e-01,
    1.999183939943416e-01,
    2.039983612187159e-01,
    2.081615930803223e-01,
]


class TestPythonMatchesCompiledEngines:
    @pytest.mark.parametrize("key", sorted(CPP_RUST_CONSTANTS))
    def test_scalar(self, regenerated, key):
        assert _close(regenerated["values"][key], CPP_RUST_CONSTANTS[key]), key

    def test_brw_weights(self, regenerated):
        got = regenerated["values"]["a_brw_weights_5_lam098"]
        assert len(got) == 5
        for g, e in zip(got, CPP_RUST_BRW_WEIGHTS):
            assert _close(g, e)
        assert sum(got) == pytest.approx(1.0, abs=1e-15)

    def test_no_constant_is_orphaned(self, regenerated):
        # Every scalar the compiled engines pin must exist in the JSON, and
        # vice versa (except the one vector-valued entry).
        json_scalars = {k for k, v in regenerated["values"].items() if not isinstance(v, list)}
        assert json_scalars == set(CPP_RUST_CONSTANTS)
