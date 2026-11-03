"""The dated-bootstrap fixture is QuantLib's, today (W13b oracle check).

Regenerates ``tests/data/dated_bootstrap_usd.txt`` with the installed
QuantLib and requires the result to equal the committed file to 1e-13 in
every number, so the oracle the Python, Rust and C++ bootstraps are tested
against cannot drift from QuantLib unnoticed. Runs in the required
``quantlib`` CI job (file name matches ``*quantlib*``); skipped elsewhere.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("QuantLib")

HERE = pathlib.Path(__file__).parent


def _numbers(text: str) -> list[list[str]]:
    return [ln.split("#", 1)[0].split() for ln in text.splitlines() if ln.split("#", 1)[0].strip()]


def test_fixture_regenerates_from_quantlib(tmp_path):
    spec = importlib.util.spec_from_file_location("gen", HERE / "oracle" / "gen_dated_bootstrap_fixture.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = tmp_path / "fixture.txt"
    gen.main(out)
    new = _numbers(out.read_text(encoding="utf-8"))
    old = _numbers((HERE / "data" / "dated_bootstrap_usd.txt").read_text(encoding="utf-8"))
    assert len(new) == len(old)
    for a, b in zip(new, old, strict=True):
        assert a[0] == b[0] and len(a) == len(b)
        for x, y in zip(a[1:], b[1:], strict=True):
            if x.lstrip("-").replace(".", "", 1).replace("e-", "", 1).isdigit():
                assert float(x) == pytest.approx(float(y), rel=1e-13, abs=1e-15)
            else:
                assert x == y
