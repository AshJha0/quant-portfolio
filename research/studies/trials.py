"""Append-only log of every training run: ``research/results/trials.jsonl``.

Why it exists: a deflated Sharpe needs the number of things tried, and until
now that number came from memory. Every training run, stress and calibration
written by the study drivers appends one JSON line here, so the trial count
of a future registration can be read off the log instead of recalled.

One line per run, never rewritten. Fields: ``timestamp_utc``, ``study``,
``kind`` (``train``, ``stress`` or ``calibration``), ``prereg``,
``prereg_sha256``, ``result`` (the result file it went with), ``git_commit``
and ``git_dirty`` (read before the result file is written), ``params``,
``metric`` (the primary metric and its standard error where one exists) and
``python``.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "research" / "results" / "trials.jsonl"
# Set QP_TRIALS_LOG to redirect the log (tests use a temporary file).
ENV = "QP_TRIALS_LOG"
FIELDS = ("timestamp_utc", "study", "kind", "prereg", "prereg_sha256", "result", "git_commit",
          "git_dirty", "params", "metric", "python")
METRIC_KEYS = ("sharpe", "sharpe_se_lo", "n_days", "active_days", "trial_count")


def log_path() -> Path:
    return Path(os.environ[ENV]) if os.environ.get(ENV) else LOG


def git_state(repo: Path = REPO) -> tuple[str | None, bool | None]:
    """(commit, dirty) of ``repo``; (None, None) outside a git checkout. The
    log itself is left out of the dirty check, or every run after the first
    would read as dirty until the log is committed."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", ".",
                                 ":(exclude)research/results/trials.jsonl"], cwd=repo, check=True,
                                capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def metric_of(payload: dict[str, Any]) -> dict[str, Any]:
    """The headline numbers of a study payload (missing keys are left out)."""
    return {k: payload[k] for k in METRIC_KEYS if k in payload}


def _plain(x: Any) -> Any:
    """JSON-safe copy: callables become their name, numpy scalars floats."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    if callable(x):
        return getattr(x, "__name__", repr(x))
    if isinstance(x, str | int | bool) or x is None:
        return x
    try:
        return float(x)
    except (TypeError, ValueError):
        return repr(x)


def append(study: str, kind: str, prereg: str, prereg_sha256: str, result: str,
           params: dict[str, Any], metric: dict[str, Any], git: tuple[str | None, bool | None] | None = None,
           path: Path | None = None) -> dict[str, Any]:
    """Append one line and return it. Opens the file in append mode only."""
    commit, dirty = git if git is not None else git_state()
    rec = {"timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"), "study": study,
           "kind": kind, "prereg": prereg, "prereg_sha256": prereg_sha256, "result": result,
           "git_commit": commit, "git_dirty": dirty, "params": _plain(params),
           "metric": _plain(metric), "python": platform.python_version()}
    out = path or log_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def read(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or log_path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def count(prereg: str, kind: str = "train", path: Path | None = None) -> int:
    """Logged runs of ``kind`` against registration file ``prereg``."""
    return sum(r["prereg"] == prereg and r["kind"] == kind for r in read(path))
