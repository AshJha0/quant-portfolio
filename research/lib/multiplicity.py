"""Programme-wide trial count and trial-Sharpe variance, read from ``trials.jsonl``.

Plain idea: a deflated Sharpe ratio asks how good the best of N tries would
look by luck. N must count every configuration the programme has run against
the data, not only the ones in the study being scored. Until Round 7 each
study passed its own count (``n_trials=3`` was hard-coded in
``fx_studies.py``), so the benchmark was too low.

Formal statement. ``n_trials`` is the sum, over distinct ``(prereg, params)``
lines of kind ``train`` or ``calibration``, of ``metric.trial_count``
(default 1), plus every line of kind ``withdrawn``. A rerun with identical
parameters is the same trial. ``trial_sr_variance`` is the sample variance of
the per-period Sharpe ratios of those lines (annualised ``metric.sharpe``
divided by ``sqrt(252)``), and of the ``sharpe_*`` cells of ``stress`` lines,
which are variants that were looked at. ``None`` when fewer than two exist.

Worked example: three train lines with trial counts 3, 1 and 7 and one
withdrawn line give N = 12 (see ``test_multiplicity.py``).

Reference: Bailey and Lopez de Prado (2014), section 3; Harvey, Liu and Zhu
(2016) on counting every test run.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "research" / "results" / "trials.jsonl"
ENV = "QP_TRIALS_LOG"
PERIODS = 252.0


@dataclass(frozen=True)
class Tally:
    n_trials: int
    trial_sr_variance: float | None
    n_sharpes: int
    by_prereg: dict[str, int]


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def programme_tally(path: Path | None = None) -> Tally:
    """Count trials and their Sharpe dispersion across the whole programme."""
    if path is None:
        path = Path(os.environ[ENV]) if os.environ.get(ENV) else LOG
    seen: set[tuple[str, str]] = set()
    n = 0
    by: dict[str, int] = {}
    sharpes: list[float] = []
    for row in _read(path):
        kind = row.get("kind")
        metric = row.get("metric") or {}
        pre = str(row.get("prereg"))
        if kind == "withdrawn":
            n += 1
            by[pre] = by.get(pre, 0) + 1
            continue
        if kind == "stress":
            sharpes += [float(v) / math.sqrt(PERIODS) for k, v in metric.items()
                        if k.startswith("sharpe_") and v is not None and math.isfinite(float(v))]
            continue
        if kind not in ("train", "calibration"):
            continue
        key = (pre, json.dumps(row.get("params"), sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        k = int(metric.get("trial_count", 1) or 1)
        n += k
        by[pre] = by.get(pre, 0) + k
        if metric.get("sharpe") is not None:
            sharpes.append(float(metric["sharpe"]) / math.sqrt(PERIODS))
    var = None
    if len(sharpes) >= 2:
        m = sum(sharpes) / len(sharpes)
        var = sum((s - m) ** 2 for s in sharpes) / (len(sharpes) - 1)
    return Tally(max(n, 1), var, len(sharpes), by)
