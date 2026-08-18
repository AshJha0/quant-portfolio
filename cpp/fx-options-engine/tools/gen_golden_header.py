#!/usr/bin/env python3
"""Generate tests/golden_vectors.hpp from the Python project's golden JSON.

The Python reference (python/fx/01-options-pricing) emits
tests/golden/golden_vectors.json; this script converts it into a C++ header
of constexpr test vectors so the GoogleTest suite cross-validates the C++
engine against the Python implementation with no runtime file I/O.

Usage:
    python3 tools/gen_golden_header.py [json_path] [header_path]

Defaults resolve relative to this repository layout.  Floats are emitted
with repr() (shortest round-trip representation), so the header reproduces
the JSON values bit-for-bit.
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_JSON = (HERE / ".." / ".." / ".." / "python" / "fx" /
                "01-options-pricing" / "tests" / "golden" /
                "golden_vectors.json").resolve()
DEFAULT_HEADER = (HERE / ".." / "tests" / "golden_vectors.hpp").resolve()

HEADER_TEMPLATE = """\
// GENERATED FILE -- do not edit by hand.
//
// Produced by tools/gen_golden_header.py from the Python reference
// project's golden vectors:
//   python/fx/01-options-pricing/tests/golden/golden_vectors.json
// {description}
// Source tolerance: {tolerance}; the C++ suite validates to 1e-9.

#pragma once

#include <array>

#include "fxopt/common.hpp"

namespace fxopt::golden {{

struct GoldenCase {{
    // inputs
    double S;
    double K;
    double T;
    double r_d;
    double r_f;
    double sigma;
    OptionType type;
    // expected outputs (from the Python fx_options implementation)
    double price;
    double delta_spot;
    double delta_fwd;
    double gamma;
    double vega;
    double theta;
    double rho_d;
    double rho_f;
}};

inline constexpr std::array<GoldenCase, {n_cases}> kCases{{{{
{rows}
}}}};

}}  // namespace fxopt::golden
"""


def fmt(x: float) -> str:
    """Shortest decimal that round-trips to the same double."""
    return repr(float(x))


def main() -> None:
    json_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_JSON
    header_path = (pathlib.Path(sys.argv[2]) if len(sys.argv) > 2
                   else DEFAULT_HEADER)
    data = json.loads(json_path.read_text())
    cases = data["cases"]
    assert len(cases) == data["n_cases"], "case count mismatch in JSON"

    rows = []
    for case in cases:
        i, o = case["inputs"], case["outputs"]
        opt = ("OptionType::Call" if i["type"] == "call"
               else "OptionType::Put")
        rows.append(
            "    {{{S}, {K}, {T}, {r_d}, {r_f}, {sigma}, {opt},\n"
            "     {price}, {delta_spot}, {delta_fwd}, {gamma},\n"
            "     {vega}, {theta}, {rho_d}, {rho_f}}},".format(
                S=fmt(i["S"]), K=fmt(i["K"]), T=fmt(i["T"]),
                r_d=fmt(i["r_d"]), r_f=fmt(i["r_f"]), sigma=fmt(i["sigma"]),
                opt=opt,
                price=fmt(o["price"]), delta_spot=fmt(o["delta_spot"]),
                delta_fwd=fmt(o["delta_fwd"]), gamma=fmt(o["gamma"]),
                vega=fmt(o["vega"]), theta=fmt(o["theta"]),
                rho_d=fmt(o["rho_d"]), rho_f=fmt(o["rho_f"])))

    header_path.write_text(HEADER_TEMPLATE.format(
        description=data.get("description", ""),
        tolerance=data.get("tolerance", ""),
        n_cases=len(cases),
        rows="\n".join(rows)))
    print(f"Wrote {header_path} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
