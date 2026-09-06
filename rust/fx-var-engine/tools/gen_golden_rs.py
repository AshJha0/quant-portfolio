#!/usr/bin/env python3
"""Emit ``tests/golden/mod.rs`` from the Python reference golden JSON.

Pipeline (see docs/ARCHITECTURE.md "golden vectors")::

    python/fx/03-var-es-engine/tests/golden/generate_golden.py
        -> python/fx/03-var-es-engine/tests/golden/golden_vectors.json
        -> (this script) rust/fx-var-engine/tests/golden/mod.rs
        -> `mod golden;` in rust/fx-var-engine/tests/test_golden_python.rs

Every constant is written with Python ``repr()`` (shortest round-trip
representation), which ``rustc`` parses back to the identical IEEE-754
``f64``, so the module is a byte-for-byte function of the JSON and
``git diff`` after a regeneration proves provenance.

Usage::

    python3 tools/gen_golden_rs.py            # (re)write tests/golden/mod.rs
    python3 tools/gen_golden_rs.py --check    # exit 1 if the module is stale
    python3 tools/gen_golden_rs.py --json path/to/golden_vectors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
REPO = ENGINE.parent.parent

DEFAULT_JSON = REPO / "python/fx/03-var-es-engine/tests/golden/golden_vectors.json"
DEFAULT_OUT = ENGINE / "tests/golden/mod.rs"


def _rs_f64(v: float) -> str:
    r = repr(float(v))
    if "e" in r:
        mant, exp = r.split("e")
        if "." not in mant:
            mant += ".0"
        return f"{mant}e{int(exp)}"
    if "." not in r:
        r += ".0"
    return r


def _rs_str(v: str) -> str:
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render(data: dict) -> str:
    lines = [
        "//! GENERATED FILE -- do not edit by hand.",
        "//!",
        f"//! Source : {data.get('source', '?')}",
        "//! Emitted: tools/gen_golden_rs.py from",
        f"//!          {DEFAULT_JSON.relative_to(REPO).as_posix()}",
        "//!",
        "//! Regenerate the JSON with the Python project's",
        "//! tests/golden/generate_golden.py, then rerun this script and commit",
        "//! both outputs together.  `tools/gen_golden_rs.py --check` fails if",
        "//! this module does not match the JSON.",
        "//!",
        f"//! {data.get('description', '')}",
        "",
        "#![allow(dead_code, clippy::excessive_precision, clippy::unreadable_literal)]",
        "",
    ]
    tol = data.get("tolerance", {})
    if tol:
        lines.append("// Consumer tolerances declared by the generator.")
        for k, v in tol.items():
            lines.append(f"pub const TOL_{k.upper()}: f64 = {_rs_f64(float(v))};")
        lines.append("")
    lines.append("// Reference values.")
    for k, v in data["values"].items():
        name = k.upper()
        if isinstance(v, list):
            body = ", ".join(_rs_f64(float(x)) for x in v)
            lines.append(f"pub const {name}: [f64; {len(v)}] = [{body}];")
        elif isinstance(v, str):
            lines.append(f"pub const {name}: &str = {_rs_str(v)};")
        else:
            lines.append(f"pub const {name}: f64 = {_rs_f64(float(v))};")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    args = ap.parse_args(argv)

    data = json.loads(args.json.read_text())
    text = render(data)
    if args.check:
        if not args.out.exists():
            print(f"MISSING: {args.out}", file=sys.stderr)
            return 1
        if args.out.read_text() != text:
            print(f"STALE: {args.out} differs from {args.json}", file=sys.stderr)
            return 1
        print(f"OK: {args.out} matches {args.json} ({len(data['values'])} values)")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {len(data['values'])} golden values to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
