#!/usr/bin/env python3
"""Emit ``tests/golden_vectors.hpp`` from the Python reference golden JSON.

Pipeline (see docs/ARCHITECTURE.md "golden vectors")::

    python/equity/03-var-es-engine/tests/golden/generate_golden.py
        -> python/equity/03-var-es-engine/tests/golden/golden_vectors.json
        -> (this script) cpp/equity-var-engine/tests/golden_vectors.hpp
        -> #include'd by cpp/equity-var-engine/tests/test_cross_language.cpp

Every constant is written with Python ``repr()`` (shortest round-trip
representation), which the C++ compiler parses back to the identical
IEEE-754 double, so the header is a byte-for-byte function of the JSON and
``git diff`` after a regeneration proves provenance.

Usage::

    python3 tools/gen_golden_header.py            # (re)write the header
    python3 tools/gen_golden_header.py --check    # exit 1 if header is stale
    python3 tools/gen_golden_header.py --json path/to/golden_vectors.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
REPO = ENGINE.parent.parent

DEFAULT_JSON = REPO / "python/equity/03-var-es-engine/tests/golden/golden_vectors.json"
DEFAULT_OUT = ENGINE / "tests/golden_vectors.hpp"
NAMESPACE = "eqvar::golden"
GUARD = "EQVAR_TESTS_GOLDEN_VECTORS_HPP"


def _c_literal(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v}.0"
    if isinstance(v, float):
        r = repr(v)
        if "e" not in r and "." not in r and "inf" not in r and "nan" not in r:
            r += ".0"
        return r
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise TypeError(f"unsupported golden value type {type(v)!r}")


def render(data: dict) -> str:
    lines = [
        "// GENERATED FILE -- do not edit by hand.",
        "//",
        f"// Source : {data.get('source', '?')}",
        "// Emitted: tools/gen_golden_header.py from",
        f"//          {DEFAULT_JSON.relative_to(REPO).as_posix()}",
        "// Regenerate the JSON with the Python project's",
        "// tests/golden/generate_golden.py, then rerun this script and commit",
        "// both outputs together.  `tools/gen_golden_header.py --check` fails",
        "// if this header does not match the JSON.",
        "//",
        f"// {data.get('description', '')}",
        "",
        f"#ifndef {GUARD}",
        f"#define {GUARD}",
        "",
        f"namespace {NAMESPACE} {{",
        "",
    ]
    tol = data.get("tolerance", {})
    if tol:
        lines.append("// Consumer tolerances declared by the generator.")
        for k, v in tol.items():
            lines.append(f"inline constexpr double tol_{k} = {_c_literal(float(v))};")
        lines.append("")
    lines.append("// Reference values.")
    for k, v in data["values"].items():
        if isinstance(v, list):
            body = ", ".join(_c_literal(float(x)) for x in v)
            lines.append(f"inline constexpr double {k}[{len(v)}] = {{{body}}};")
        elif isinstance(v, str):
            lines.append(f"inline constexpr const char* {k} = {_c_literal(v)};")
        else:
            lines.append(f"inline constexpr double {k} = {_c_literal(float(v))};")
    lines += ["", f"}}  // namespace {NAMESPACE}", "", f"#endif  // {GUARD}", ""]
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
    args.out.write_text(text)
    print(f"wrote {len(data['values'])} golden values to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
