"""Aggregate executed test results from CI artefacts into ``counts.json``.

Inputs, all under one directory as laid down by ``actions/download-artifact``:

* ``junit-py-<slug>-<pyver>/junit.xml``: pytest JUnit XML, one per project
  and Python version.
* ``junit-cpp-<engine>-<compiler>/junit.xml``: ``ctest --output-junit``.
* ``rust-<crate>/cargo-test.log``: ``cargo test`` output. Every
  ``test result: ok. N passed; M failed; K ignored`` line is summed.

Output: per language and per project, the number of tests that were executed
and passed, failed, errored or were skipped. The headline Python counts use
the reference Python version (``--ref-python``). The headline C++ counts use
the reference compiler (``--ref-cxx``).

Usage::

    python .github/scripts/aggregate_counts.py ARTEFACT_DIR -o counts.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

RUST_LINE = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored")
KEYS = ("passed", "failed", "errors", "skipped")


def _zero() -> dict[str, int]:
    return dict.fromkeys(KEYS, 0)


def parse_junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    c = _zero()
    for s in suites:
        cases = list(s.iter("testcase"))
        for case in cases:
            kinds = {child.tag for child in case}
            if "failure" in kinds:
                c["failed"] += 1
            elif "error" in kinds:
                c["errors"] += 1
            elif "skipped" in kinds:
                c["skipped"] += 1
            else:
                c["passed"] += 1
        if not cases and int(s.get("errors", 0) or 0):
            c["errors"] += int(s.get("errors"))
    return c


def parse_cargo(path: Path) -> dict[str, int]:
    c = _zero()
    for m in RUST_LINE.finditer(path.read_text(encoding="utf-8", errors="replace")):
        c["passed"] += int(m.group(1))
        c["failed"] += int(m.group(2))
        c["skipped"] += int(m.group(3))
    return c


def _total(section: dict[str, dict[str, int]]) -> dict[str, int]:
    t = _zero()
    for c in section.values():
        for k in KEYS:
            t[k] += c[k]
    return t


def aggregate(art: Path, ref_python: str, ref_cxx: str) -> dict:
    out: dict = {
        "schema": 1,
        "reference": {"python": ref_python, "cxx": ref_cxx},
        "python": {}, "python_by_version": {},
        "cpp": {}, "cpp_by_compiler": {},
        "rust": {},
    }
    for d in sorted(p for p in art.iterdir() if p.is_dir()):
        name = d.name
        if name.startswith("junit-py-"):
            slug, ver = name.removeprefix("junit-py-").rsplit("-", 1)
            c = parse_junit(d / "junit.xml")
            out["python_by_version"].setdefault(slug, {})[ver] = c
            if ver == ref_python:
                out["python"][slug] = c
        elif name.startswith("junit-cpp-"):
            engine, cxx = name.removeprefix("junit-cpp-").rsplit("-", 1)
            c = parse_junit(d / "junit.xml")
            out["cpp_by_compiler"].setdefault(engine, {})[cxx] = c
            if cxx == ref_cxx:
                out["cpp"][engine] = c
        elif name.startswith("rust-"):
            out["rust"][name.removeprefix("rust-")] = parse_cargo(d / "cargo-test.log")
    out["totals"] = {k: _total(out[k]) for k in ("python", "cpp", "rust")}
    out["totals"]["all_passed"] = sum(out["totals"][k]["passed"] for k in ("python", "cpp", "rust"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate CI test artefacts into counts.json")
    ap.add_argument("artefacts", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("counts.json"))
    ap.add_argument("--ref-python", default="3.12")
    ap.add_argument("--ref-cxx", default="gcc")
    args = ap.parse_args(argv)
    counts = aggregate(args.artefacts, args.ref_python, args.ref_cxx)
    args.output.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    t = counts["totals"]
    for k in ("python", "cpp", "rust"):
        print(f"{k:7s} {t[k]}")
    print(f"all passed: {t['all_passed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
