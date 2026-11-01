"""Compare CI benchmark p50 latencies with the committed bench/results.json.

CI runners are shared virtual machines. Their numbers are not the published
numbers in the engine READMEs and are not comparable to them in absolute
terms. This script only catches large regressions:

* A kernel FAILS when its committed p50 is above 1 microsecond (1000 ns) and
  the CI p50 is more than 2.0 times the committed p50.
* Every other difference is reported, never failed: kernels at or below
  1 us (timer and runner noise dominate there), speed-ups, kernels present
  on only one side, and engines with no committed file.

Input schemas. Each results file is a JSON list of records (a dict holding
such a list under ``results``, ``benchmarks`` or ``kernels`` is accepted
too). A record needs ``kernel`` and ``p50``; ``unit`` is ``ns`` when absent
and may be ``ns``, ``us``, ``ms`` or ``s``. The C++ files carry a ``label``
(``before`` / ``after``) and several runs per kernel (different compilers and
dates); the Rust files carry no label. The baseline for a kernel is the
median p50 over its records, restricted to ``label == "after"`` when the file
has any ``after`` record. The same rule is applied to the CI file.

Usage::

    python .github/scripts/bench_compare.py \\
        --pair cpp/equity-options-engine cpp/equity-options-engine/bench/results.json ci/eqopt.json \\
        --pair rust/fx-var-engine rust/fx-var-engine/bench/results.json ci/fxvar.json \\
        --summary summary.md --json comparison.json

Exit status: 0 when no kernel fails, 1 when one does, 2 on unusable input
(a CI results file missing or without any timed kernel).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

FACTOR = 2.0
FLOOR_NS = 1000.0
UNIT_NS = {"ns": 1.0, "us": 1e3, "µs": 1e3, "ms": 1e6, "s": 1e9}


class BenchInputError(ValueError):
    """A results file cannot be read as benchmark records."""


def _records(data: object, path: Path) -> list[dict]:
    if isinstance(data, dict):
        for key in ("results", "benchmarks", "kernels"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise BenchInputError(f"{path}: expected a JSON list of records")
    return [r for r in data if isinstance(r, dict)]


def load_p50(path: Path) -> dict[str, float]:
    """Kernel name -> p50 in nanoseconds, per the selection rule above."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchInputError(f"{path}: not valid JSON ({exc})") from exc
    recs = _records(data, path)
    if any(r.get("label") == "after" for r in recs):
        recs = [r for r in recs if r.get("label") == "after"]
    per: dict[str, list[float]] = {}
    for r in recs:
        kernel, p50 = r.get("kernel"), r.get("p50")
        if not isinstance(kernel, str) or isinstance(p50, bool) or not isinstance(p50, (int, float)):
            continue
        unit = str(r.get("unit", "ns"))
        if unit not in UNIT_NS:
            raise BenchInputError(f"{path}: kernel {kernel!r} has unknown unit {unit!r}")
        per.setdefault(kernel, []).append(float(p50) * UNIT_NS[unit])
    return {k: statistics.median(v) for k, v in per.items()}


@dataclass
class Row:
    engine: str
    kernel: str
    committed_ns: float | None
    ci_ns: float | None
    ratio: float | None
    status: str  # fail | ok | report | new | missing


def compare(engine: str, committed: dict[str, float], ci: dict[str, float]) -> list[Row]:
    rows = []
    for k in sorted(set(committed) | set(ci)):
        c, n = committed.get(k), ci.get(k)
        if c is None:
            rows.append(Row(engine, k, None, n, None, "new"))
        elif n is None:
            rows.append(Row(engine, k, c, None, None, "missing"))
        else:
            ratio = n / c if c > 0 else float("inf")
            if c <= FLOOR_NS:
                status = "report"
            else:
                status = "fail" if ratio > FACTOR else "ok"
            rows.append(Row(engine, k, c, n, ratio, status))
    return rows


def _fmt(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x / 1e3:,.2f} us" if x >= 1e4 else f"{x:,.1f} ns"


def markdown(rows: list[Row], notes: list[str]) -> str:
    fails = [r for r in rows if r.status == "fail"]
    out = [
        "## Benchmarks: CI runner p50 vs committed bench/results.json",
        "",
        "CI runner numbers are not the published numbers. A kernel fails only when its "
        f"committed p50 is above {FLOOR_NS / 1e3:g} us and the CI p50 is more than "
        f"{FACTOR:g}x the committed value. Everything else is report-only.",
        "",
        f"**{len(fails)} failing kernel(s)** out of {len(rows)} compared.",
        "",
    ]
    out += [f"- {n}" for n in notes]
    if notes:
        out.append("")
    out += ["| engine | kernel | committed p50 | CI p50 | ratio | status |",
            "|---|---|---:|---:|---:|---|"]
    for r in rows:
        ratio = "-" if r.ratio is None else f"{r.ratio:.2f}x"
        status = "**FAIL**" if r.status == "fail" else r.status
        kernel = r.kernel.replace("|", "\\|")
        out.append(f"| {r.engine} | {kernel} | {_fmt(r.committed_ns)} | {_fmt(r.ci_ns)} | {ratio} | {status} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pair", nargs=3, action="append", required=True,
                    metavar=("ENGINE", "COMMITTED", "CI"), help="engine label, committed JSON, CI JSON")
    ap.add_argument("--summary", type=Path, help="write the markdown summary here")
    ap.add_argument("--json", type=Path, help="write the per-kernel comparison here")
    args = ap.parse_args(argv)

    rows: list[Row] = []
    notes: list[str] = []
    try:
        for engine, committed_path, ci_path in args.pair:
            ci_file = Path(ci_path)
            if not ci_file.is_file():
                raise BenchInputError(f"{engine}: CI results file {ci_path} does not exist")
            ci = load_p50(ci_file)
            if not ci:
                raise BenchInputError(f"{engine}: CI results file {ci_path} holds no timed kernel")
            committed_file = Path(committed_path)
            if not committed_file.is_file():
                notes.append(f"{engine}: no committed results file, report-only")
                committed = {}
            else:
                committed = load_p50(committed_file)
            rows += compare(engine, committed, ci)
    except BenchInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    md = markdown(rows, notes)
    print(md)
    if args.summary:
        args.summary.write_text(md, encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n", encoding="utf-8")
    return 1 if any(r.status == "fail" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
