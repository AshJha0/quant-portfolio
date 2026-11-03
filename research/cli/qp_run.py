"""``qp-run sweep`` and ``qp-run trials {query,rerun}`` (W23 part 2).

Plain idea: every parameter combination a researcher tries is a trial, and
a deflated Sharpe is only honest if every trial is counted. ``sweep`` runs a
grid and writes one ``trials.jsonl`` row per point, so nothing tried goes
unlogged. Each row also records the SHA-256 of every input file and of
``uv.lock``, so a later reader can tell whether a rerun used the same data
and the same library versions. ``trials rerun`` re-executes one logged row
and compares its metrics to 1e-12; the nightly job runs it on a random row.

Usage::

    python research/cli/qp_run.py sweep spec.json [--log PATH] [--dry-run]
    python research/cli/qp_run.py trials query [--study S] [--kind K]
        [--prereg P] [--json] [--log PATH]
    python research/cli/qp_run.py trials rerun (--index I | --random SEED)
        [--tol 1e-12] [--log PATH]

A sweep spec is JSON::

    {"study": "name", "fn": "module.path:function", "kind": "sweep",
     "prereg": "exploratory", "datasets": ["data/raw/x.csv"],
     "fixed": {"cost_bp": 1.0}, "grid": {"lookback": [21, 63, 126]}}

``fn`` is called as ``fn(**fixed, **point)`` and must return a dict of
numbers (the metrics). ``prereg`` defaults to ``exploratory``: a sweep is
never confirmatory, and a row logged against a signed registration's path
would be refused by ``prereg.py check`` only if it wrote a result file,
which a sweep does not. Exit codes: 0 ok, 1 rerun mismatch, 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import platform
import random
import sys
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _p in ("research/studies", "research/lib", "research/tools", "research"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import trials  # noqa: E402

LOCKFILE = REPO / "uv.lock"
SWEEP_KINDS = ("sweep", "train")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grid_points(grid: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    """Cartesian product in sorted key order, so row order is reproducible."""
    keys = sorted(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals, strict=True))


def resolve(fn: str) -> Callable[..., dict[str, Any]]:
    mod, _, name = fn.partition(":")
    if not mod or not name:
        raise ValueError(f"fn must be 'module:function', got {fn!r}")
    return getattr(importlib.import_module(mod), name)


def make_row(study: str, kind: str, prereg: str, fn: str, params: dict[str, Any],
             metric: dict[str, Any], datasets: dict[str, str], lock_sha: str | None,
             git: tuple[str | None, bool | None]) -> dict[str, Any]:
    """A ``trials.jsonl`` row: the W6 fields plus ``fn``, ``datasets`` and
    ``lockfile_sha256`` (W23). ``prereg_sha256`` is empty for exploratory rows."""
    commit, dirty = git
    pre_sha = ""
    pre_path = REPO / prereg
    if prereg != "exploratory" and pre_path.is_file():
        import prereg as pr  # research/tools

        pre_sha = pr.registration_hash(pre_path.read_text(encoding="utf-8"))
    return {"timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"), "study": study,
            "kind": kind, "prereg": prereg, "prereg_sha256": pre_sha, "result": "",
            "git_commit": commit, "git_dirty": dirty, "params": trials._plain(params),
            "metric": trials._plain(metric), "python": platform.python_version(), "fn": fn,
            "datasets": datasets, "lockfile_sha256": lock_sha}


def sweep(spec: dict[str, Any], log: Path | None = None, dry_run: bool = False,
          git: tuple[str | None, bool | None] | None = None) -> list[dict[str, Any]]:
    for k in ("study", "fn", "grid"):
        if k not in spec:
            raise ValueError(f"spec is missing {k!r}")
    kind = spec.get("kind", "sweep")
    if kind not in SWEEP_KINDS:
        raise ValueError(f"kind must be one of {SWEEP_KINDS}")
    fixed = dict(spec.get("fixed", {}))
    overlap = set(fixed) & set(spec["grid"])
    if overlap:
        raise ValueError(f"parameters both fixed and swept: {sorted(overlap)}")
    datasets = {d: sha256_file(REPO / d) for d in spec.get("datasets", [])}
    lock_sha = sha256_file(LOCKFILE) if LOCKFILE.exists() else None
    fn = resolve(spec["fn"])
    g = git if git is not None else trials.git_state()
    rows = []
    for point in grid_points(spec["grid"]):
        params = {**fixed, **point}
        metric = fn(**params)
        row = make_row(spec["study"], kind, spec.get("prereg", "exploratory"), spec["fn"],
                       params, metric, datasets, lock_sha, g)
        rows.append(row)
        if not dry_run:
            out = log or trials.log_path()
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def query(rows: list[dict[str, Any]], study: str | None = None, kind: str | None = None,
          prereg: str | None = None) -> list[dict[str, Any]]:
    return [r for r in rows if (study is None or r.get("study") == study)
            and (kind is None or r.get("kind") == kind)
            and (prereg is None or r.get("prereg") == prereg)]


def compare(old: dict[str, Any], new: dict[str, Any], tol: float) -> list[str]:
    """Keys whose values differ by more than ``tol`` (absolute), or are missing."""
    bad = []
    for k, v in old.items():
        w = new.get(k)
        if isinstance(v, int | float) and isinstance(w, int | float):
            if math.isnan(v) and math.isnan(w):
                continue
            if not abs(float(v) - float(w)) <= tol:
                bad.append(f"{k}: {v!r} -> {w!r}")
        elif v != w:
            bad.append(f"{k}: {v!r} -> {w!r}")
    return bad


def rerun(row: dict[str, Any], tol: float = 1e-12) -> list[str]:
    """Re-execute a sweep row and list the metric differences (empty = same).

    Refuses (ValueError) rows without ``fn`` and rows whose recorded datasets
    no longer hash to the logged value: a different number from different
    data proves nothing about reproducibility."""
    if not row.get("fn"):
        raise ValueError("row has no 'fn'; only qp-run sweep rows can be rerun")
    for path, sha in (row.get("datasets") or {}).items():
        now = sha256_file(REPO / path)
        if now != sha:
            raise ValueError(f"dataset {path} changed: {sha} -> {now}")
    new = trials._plain(resolve(row["fn"])(**row["params"]))
    return compare(row["metric"], new, tol)


def _table(rows: list[dict[str, Any]]) -> str:
    lines = ["idx  timestamp_utc              study                kind   metric"]
    for i, r in enumerate(rows):
        m = r.get("metric", {})
        head = ", ".join(f"{k}={m[k]:.6g}" if isinstance(m[k], float) else f"{k}={m[k]}"
                         for k in sorted(m)[:4])
        lines.append(f"{i:<4} {r.get('timestamp_utc', ''):<26} {r.get('study', ''):<20} "
                     f"{r.get('kind', ''):<6} {head}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="qp-run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sw = sub.add_parser("sweep")
    sw.add_argument("spec", type=Path)
    sw.add_argument("--log", type=Path)
    sw.add_argument("--dry-run", action="store_true")
    tr = sub.add_parser("trials")
    tsub = tr.add_subparsers(dest="tcmd", required=True)
    q = tsub.add_parser("query")
    q.add_argument("--study")
    q.add_argument("--kind")
    q.add_argument("--prereg")
    q.add_argument("--json", action="store_true")
    q.add_argument("--log", type=Path)
    rr = tsub.add_parser("rerun")
    pick = rr.add_mutually_exclusive_group(required=True)
    pick.add_argument("--index", type=int)
    pick.add_argument("--random", type=int, metavar="SEED")
    rr.add_argument("--tol", type=float, default=1e-12)
    rr.add_argument("--log", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "sweep":
            rows = sweep(json.loads(a.spec.read_text(encoding="utf-8")), a.log, a.dry_run)
            print(_table(rows))
            return 0
        rows = trials.read(a.log)
        if a.tcmd == "query":
            hit = query(rows, a.study, a.kind, a.prereg)
            print(json.dumps(hit, indent=2, sort_keys=True) if a.json else _table(hit))
            return 0
        cand = [r for r in rows if r.get("fn")]
        if not cand:
            print("no rerunnable rows", file=sys.stderr)
            return 2
        row = cand[a.index] if a.index is not None else random.Random(a.random).choice(cand)
        diff = rerun(row, a.tol)
        if diff:
            print("MISMATCH\n" + "\n".join(diff))
            return 1
        print(f"OK {row['study']} {row['params']} reproduced to {a.tol:g}")
        return 0
    except (ValueError, KeyError, IndexError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
