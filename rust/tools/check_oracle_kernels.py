"""Check the Rust oracle-kernel list in ``rust/ORACLE_KERNELS.md`` (plan W20 target 4).

Static checks (default): every row names an existing crate, test file and
``#[test] fn``; the test file calls every listed Rust entry point; the
generated constants module exists and names the oracle source, which exists.

``--run`` also runs each listed test once with cargo and requires exactly
one passing test per row. Exit status 0 when everything holds, 1 otherwise.

Usage::

    python rust/tools/check_oracle_kernels.py            # static
    python rust/tools/check_oracle_kernels.py --run      # static + cargo
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

RUST = Path(__file__).resolve().parents[1]
REPO = RUST.parent
DOC = RUST / "ORACLE_KERNELS.md"
COLUMNS = ("id", "Kernel", "Crate", "Test file", "Test", "Rust entry points",
           "Constants module", "Oracle source")


@dataclass(frozen=True)
class Row:
    id: str
    crate: str
    test_file: str
    test: str
    symbols: tuple[str, ...]
    constants: str
    oracle: str


def parse(text: str) -> list[Row]:
    """Rows of the first Markdown table whose header is ``COLUMNS``."""
    lines = [ln.strip() for ln in text.splitlines()]
    header = "| " + " | ".join(COLUMNS) + " |"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise SystemExit(f"{DOC.name}: kernel table header not found") from exc
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(COLUMNS):
            raise SystemExit(f"{DOC.name}: bad row {ln!r}")
        rid, _kernel, crate, tfile, test, syms, const, oracle = cells
        rows.append(Row(rid, crate, tfile, test,
                        tuple(s.strip() for s in syms.split(",") if s.strip()), const, oracle))
    if not rows:
        raise SystemExit(f"{DOC.name}: kernel table is empty")
    return rows


def check_static(rows: list[Row]) -> list[str]:
    errors: list[str] = []
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")
    for r in rows:
        crate = RUST / r.crate
        tf = crate / r.test_file
        if not (crate / "Cargo.toml").is_file():
            errors.append(f"{r.id}: crate {r.crate} not found")
            continue
        if not tf.is_file():
            errors.append(f"{r.id}: test file {r.test_file} not found")
            continue
        src = tf.read_text(encoding="utf-8")
        if not re.search(r"#\[test\]\s*(?:#\[[^\]]*\]\s*)*fn\s+" + re.escape(r.test) + r"\s*\(", src):
            errors.append(f"{r.id}: no #[test] fn {r.test} in {r.test_file}")
        for s in r.symbols:
            if not re.search(r"\b" + re.escape(s) + r"\s*\(", src):
                errors.append(f"{r.id}: {r.test_file} never calls {s}(")
        const = crate / r.constants
        oracle = REPO / r.oracle
        if not oracle.is_file():
            errors.append(f"{r.id}: oracle source {r.oracle} not found")
        if not const.is_file():
            errors.append(f"{r.id}: constants module {r.constants} not found")
        elif Path(r.oracle).name not in const.read_text(encoding="utf-8")[:2000]:
            errors.append(f"{r.id}: {r.constants} header does not name {Path(r.oracle).name}")
    return errors


def package_name(crate_dir: Path) -> str:
    """The ``[package] name`` of a crate (it can differ from the directory)."""
    m = re.search(r'^\[package\][^\[]*?^name\s*=\s*"([^"]+)"',
                  (crate_dir / "Cargo.toml").read_text(encoding="utf-8"), re.M | re.S)
    if m is None:
        raise SystemExit(f"{crate_dir}/Cargo.toml: no package name")
    return m.group(1)


def run_tests(rows: list[Row]) -> list[str]:
    errors: list[str] = []
    for r in rows:
        cmd = ["cargo", "test", "--release", "--locked", "-p", package_name(RUST / r.crate),
               "--test", Path(r.test_file).stem, "--", "--exact", r.test]
        proc = subprocess.run(cmd, cwd=RUST, capture_output=True, text=True, check=False)
        out = proc.stdout + proc.stderr
        if proc.returncode != 0 or "test result: ok. 1 passed" not in out:
            errors.append(f"{r.id}: `{' '.join(cmd)}` did not pass exactly one test\n{out[-2000:]}")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true", help="also run each test with cargo")
    args = ap.parse_args(argv)
    rows = parse(DOC.read_text(encoding="utf-8"))
    errors = check_static(rows)
    if args.run and not errors:
        errors = run_tests(rows)
    for e in errors:
        print(f"ORACLE_KERNELS: {e}", file=sys.stderr)
    print(f"ORACLE_KERNELS: {len(rows)} rows, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
