"""Golden vectors: the committed JSON must reproduce exactly (1e-10)."""

import json
from pathlib import Path

import pytest

from eq_options import bs_greeks

GOLDEN = Path(__file__).parent / "golden" / "golden_vectors.json"


def _load() -> list[dict]:
    return json.loads(GOLDEN.read_text())["cases"]


def test_golden_file_exists_and_has_enough_cases() -> None:
    cases = _load()
    assert len(cases) >= 30
    for case in cases:
        assert set(case["inputs"]) == {"S", "K", "T", "r", "q", "sigma", "type"}
        assert set(case["outputs"]) == {"price", "delta", "gamma", "vega",
                                        "theta", "rho"}


@pytest.mark.parametrize("case", _load(),
                         ids=lambda c: (f"{c['inputs']['type']}_S{c['inputs']['S']}"
                                        f"_K{c['inputs']['K']}_T{c['inputs']['T']}"))
def test_golden_vectors_reproduce_to_1e10(case: dict) -> None:
    inp = case["inputs"]
    g = bs_greeks(inp["S"], inp["K"], inp["T"], inp["r"], inp["sigma"],
                  inp["q"], inp["type"])
    for name, expected in case["outputs"].items():
        assert getattr(g, name) == pytest.approx(expected, abs=1e-10), name
