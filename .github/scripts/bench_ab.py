"""Same-runner A/B latency gate: base commit against head commit.

The bench job builds the base commit (the pull request's target, or the
previous commit on a push) and the head commit on one runner, then runs
their benches in alternating rounds (base, head, base, head, ...), so both
sides see the same machine, the same load and the same thermal state. No
number measured on another machine is involved.

Each round writes one results file per engine and side, in the format of
the shared harness (``cpp/core/include/qp/bench/latency.hpp`` and
``rust/core/src/bench.rs``): a JSON list of records with ``kernel``,
``p10``, ``p50`` and ``p90`` in ns per call (``unit`` defaults to ``ns``;
a dict holding the list under ``results``, ``benchmarks`` or ``kernels`` is
accepted too). Older files without ``p10``/``p90`` are read with the band
collapsed to the median.

Per kernel and side the script takes the median over rounds of p10, p50 and
p90. A kernel FAILS only when both hold:

* the head median is more than 25% above the base median, and
* the bands do not overlap: head p10 is above base p90.

Everything else is reported, never failed: smaller or overlapping changes,
speed-ups, and kernels present on one side only (new or removed kernels).

Usage::

    python .github/scripts/bench_ab.py \\
        --engine cpp/equity-options-engine \\
            --base r1/base/cpp-equity-options-engine.json r2/base/... \\
            --head r1/head/cpp-equity-options-engine.json r2/head/... \\
        [--engine ...] --summary summary.md --json ab.json

Exit status: 0 when no kernel fails, 1 when one does, 2 on unusable input
(a head file missing or holding no timed kernel).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

THRESHOLD = 1.25
UNIT_NS = {"ns": 1.0, "us": 1e3, "µs": 1e3, "ms": 1e6, "s": 1e9}


class BenchInputError(ValueError):
    """A results file cannot be read as benchmark records."""


@dataclass(frozen=True)
class Band:
    p10: float
    p50: float
    p90: float


@dataclass(frozen=True)
class Row:
    engine: str
    kernel: str
    base: Band | None
    head: Band | None
    ratio: float | None
    status: str  # FAIL, ACCEPTED, slower, faster, same, new, removed


def _records(data: object, path: Path) -> list[dict]:
    if isinstance(data, dict):
        for key in ("results", "benchmarks", "kernels"):
            if isinstance(data.get(key), list):
                return data[key]
        raise BenchInputError(f"{path}: no record list")
    if not isinstance(data, list):
        raise BenchInputError(f"{path}: not a JSON list")
    return data


def load_bands(path: Path) -> dict[str, Band]:
    """Kernel -> (p10, p50, p90) in ns from one results file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise BenchInputError(f"{path}: {e}") from e
    out: dict[str, Band] = {}
    for rec in _records(data, path):
        if not isinstance(rec, dict) or "kernel" not in rec or "p50" not in rec:
            continue
        scale = UNIT_NS.get(str(rec.get("unit", "ns")))
        if scale is None:
            raise BenchInputError(f"{path}: unknown unit {rec.get('unit')!r}")
        p50 = float(rec["p50"]) * scale
        p10 = float(rec.get("p10", rec["p50"])) * scale
        p90 = float(rec.get("p90", rec["p50"])) * scale
        out[str(rec["kernel"])] = Band(p10, p50, p90)
    return out


def pool(rounds: list[dict[str, Band]]) -> dict[str, Band]:
    """Median over rounds of each percentile, per kernel."""
    kernels = sorted({k for r in rounds for k in r})
    out = {}
    for k in kernels:
        bands = [r[k] for r in rounds if k in r]
        out[k] = Band(
            statistics.median(b.p10 for b in bands),
            statistics.median(b.p50 for b in bands),
            statistics.median(b.p90 for b in bands),
        )
    return out


def classify(base: Band | None, head: Band | None) -> tuple[float | None, str]:
    if base is None:
        return None, "new"
    if head is None:
        return None, "removed"
    if base.p50 <= 0.0:
        return None, "same"
    ratio = head.p50 / base.p50
    if ratio > THRESHOLD and head.p10 > base.p90:
        return ratio, "FAIL"
    if ratio < 1.0 / THRESHOLD and head.p90 < base.p10:
        return ratio, "faster"
    if ratio > 1.0:
        return ratio, "slower" if ratio > THRESHOLD else "same"
    return ratio, "same"


def compare(engine: str, base: dict[str, Band], head: dict[str, Band]) -> list[Row]:
    rows = []
    for k in sorted(set(base) | set(head)):
        ratio, status = classify(base.get(k), head.get(k))
        rows.append(Row(engine, k, base.get(k), head.get(k), ratio, status))
    return rows


def _fmt(b: Band | None) -> str:
    return "-" if b is None else f"{b.p50:,.1f} [{b.p10:,.1f}, {b.p90:,.1f}]"


def markdown(rows: list[Row], notes: list[str], n_rounds: int) -> str:
    lines = [
        "## Benchmarks: head vs base on the same runner",
        "",
        f"{n_rounds} interleaved rounds; median over rounds of each percentile, ns per call, "
        "shown as p50 [p10, p90]. FAIL = head median more than 25% above base and "
        "head p10 above base p90. ACCEPTED = a listed, deliberate slowdown.",
        "",
        "| engine | kernel | base | head | head/base | status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        ratio = "-" if r.ratio is None else f"{r.ratio:.2f}"
        lines.append(
            f"| {r.engine} | {r.kernel} | {_fmt(r.base)} | {_fmt(r.head)} | {ratio} | {r.status} |"
        )
    lines += ["", *[f"- {n}" for n in notes]]
    return "\n".join(lines) + "\n"


def load_accepted(path: Path) -> dict[tuple[str, str], dict]:
    """Rows a pull request may let through with the ``bench-accepted`` label.

    Each entry names one engine and kernel exactly, gives a ceiling on the
    head/base ratio and says why the slowdown is deliberate. A row not listed,
    or slower than its ceiling, still fails.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict] = {}
    for e in data["accepted"]:
        if not (e.get("engine") and e.get("kernel") and e.get("reason")):
            raise BenchInputError(f"{path}: every entry needs engine, kernel and reason")
        out[(e["engine"], e["kernel"])] = e
    return out


def apply_accepted(rows: list[Row], accepted: dict[tuple[str, str], dict]) -> list[Row]:
    """Turn FAIL into ACCEPTED for listed rows within their ratio ceiling."""
    result = []
    for r in rows:
        e = accepted.get((r.engine, r.kernel))
        if (
            r.status == "FAIL"
            and e is not None
            and r.ratio is not None
            and r.ratio <= float(e["max_ratio"])
        ):
            r = Row(r.engine, r.kernel, r.base, r.head, r.ratio, "ACCEPTED")
        result.append(r)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--engine", action="append", required=True)
    ap.add_argument("--base", action="append", nargs="+", default=[])
    ap.add_argument("--head", action="append", nargs="+", default=[])
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument(
        "--accepted",
        type=Path,
        help="JSON list of deliberate slowdowns; applied only with --allow-accepted",
    )
    ap.add_argument("--allow-accepted", action="store_true")
    a = ap.parse_args(argv)
    if not (len(a.engine) == len(a.base) == len(a.head)):
        ap.error("give one --base and one --head list per --engine")
    rows: list[Row] = []
    notes: list[str] = []
    n_rounds = 0
    try:
        for engine, base_paths, head_paths in zip(a.engine, a.base, a.head, strict=True):
            heads = []
            for p in map(Path, head_paths):
                if not p.exists():
                    raise BenchInputError(f"{engine}: head results file {p} does not exist")
                heads.append(load_bands(p))
            if not any(heads):
                raise BenchInputError(f"{engine}: head results hold no timed kernel")
            bases = [load_bands(Path(p)) for p in base_paths if Path(p).exists()]
            if not bases:
                notes.append(f"{engine}: no base results (bench missing on base), report-only")
            n_rounds = max(n_rounds, len(heads))
            rows += compare(engine, pool(bases) if bases else {}, pool(heads))
        if a.allow_accepted and a.accepted:
            rows = apply_accepted(rows, load_accepted(a.accepted))
            if any(r.status == "ACCEPTED" for r in rows):
                notes.append(
                    f"ACCEPTED rows are deliberate slowdowns listed in {a.accepted}, "
                    "allowed by the bench-accepted label; see docs/BENCHMARKS.md."
                )
    except BenchInputError as e:
        print(f"bench_ab: {e}", file=sys.stderr)
        return 2
    text = markdown(rows, notes, n_rounds)
    print(text)
    if a.summary:
        a.summary.write_text(text, encoding="utf-8")
    if a.json:
        a.json.write_text(json.dumps([asdict(r) for r in rows], indent=1), encoding="utf-8")
    return 1 if any(r.status == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
