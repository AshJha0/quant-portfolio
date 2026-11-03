"""Replay the live recordings through the C++ and Rust handlers and check
them against the Python reference book.

The rule (the same one cpp/ull/feed's fault-injection tests use): whenever a
handler reports *not stale*, its book equals the reference book at the same
update id (Binance) or record index (Coinbase level2). Each run must also
check at least half the records and end fresh.

Toolchains: C++ needs g++ (through WSL on Windows), Rust needs cargo. A
missing toolchain skips that half; set QP_RECORDER_REQUIRE_TOOLCHAINS=1 (CI)
to turn the skip into a failure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.recorder.qp_recorder.compare import compare, load, reference
from tools.recorder.qp_recorder.replay import Runner, build_cpp, build_rust

REC_DIR = Path(__file__).resolve().parents[1] / "recordings"
MANIFEST = json.loads((REC_DIR / "manifest.json").read_text())["recordings"]
REQUIRE = os.environ.get("QP_RECORDER_REQUIRE_TOOLCHAINS") == "1"

@pytest.fixture(scope="session")
def runners(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Runner | None]:
    d = tmp_path_factory.mktemp("replay-build")
    return {"cpp": build_cpp(d), "rust": build_rust(d)}


@pytest.fixture(scope="session")
def unpacked(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("recordings")
    out = {}
    for e in MANIFEST:
        p = d / e["file"].removesuffix(".gz")
        p.write_bytes(load(REC_DIR / e["file"]))
        out[e["file"]] = p
    return out


CASES = [
    pytest.param(lang, e, id=f"{lang}-{e['file'].split('.')[0]}")
    for e in MANIFEST
    for lang in ("cpp", "rust")
]


@pytest.mark.parametrize(("lang", "entry"), CASES)
def test_handler_book_equals_reference_whenever_fresh(
    lang: str, entry: dict, runners: dict[str, Runner | None], unpacked: dict[str, Path]
) -> None:
    run = runners[lang]
    if run is None:
        if REQUIRE:
            pytest.fail(f"{lang} toolchain missing")
        pytest.skip(f"{lang} toolchain missing")
    path = unpacked[entry["file"]]
    ref = reference(entry["venue"], path.read_bytes(), entry["symbol"])
    res = compare(entry["venue"], run(entry["venue"], path, entry["symbol"]), ref)
    assert res.records == entry["records"]
    assert res.mismatches == [], (
        f"{len(res.mismatches)} fresh states differ, first at record {res.mismatches[0]}"
    )
    assert res.checked >= res.records // 2
    assert res.final_fresh
