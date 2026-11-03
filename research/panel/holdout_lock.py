"""The 2016-2026 hold-out lock for the W23 panel studies, enforced in code.

Plain idea: a hold-out only tests anything if nobody looks at it while the
study is being built. A rule in a document is easy to break by accident: one
plot of the full sample and the hold-out is spent. So every panel function
passes its inputs through :meth:`HoldoutLock.guard`, which raises if any row
is dated on or after 2016-01-01 while the lock is closed.

Formal statement. The lock opens only for a registration that

1. lives in ``research/prereg/`` and has a dated file name
   (``YYYY-MM-DD-<name>.md``): a NEW registration, never an edited one;
2. is signed with the SHA-256 of its current text (``research/tools/prereg.py``);
3. names the hold-out window ``2016-01-01`` to ``2026-12-31`` in its body;
4. is not withdrawn; and
5. has not opened the lock before. Each opening is appended to the ledger
   ``research/panel/holdout_ledger.jsonl`` with the registration hash, so a
   second look with the same registration is refused.

Worked example (``tests/test_holdout_lock.py``): a 2015-12-31 row passes the
closed lock; a 2016-01-01 row raises :class:`HoldoutLocked`; an unsigned
registration raises; a signed one opens the lock once and the second
attempt raises.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "research" / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "research" / "tools"))

import prereg  # noqa: E402

HOLDOUT_START = pd.Timestamp("2016-01-01")
HOLDOUT_END = pd.Timestamp("2026-12-31")
LEDGER = Path(__file__).resolve().parent / "holdout_ledger.jsonl"
_DATED = re.compile(r"^\d{4}-\d{2}-\d{2}-[\w.-]+\.md$")


class HoldoutLocked(RuntimeError):
    """Data on or after the hold-out start reached a panel function while locked."""


@dataclass(frozen=True)
class HoldoutLock:
    registration: str | None = None  # SHA-256 of the registration that opened it

    @property
    def is_open(self) -> bool:
        return self.registration is not None

    def guard(self, obj: pd.Series | pd.DataFrame, what: str = "input") -> pd.Series | pd.DataFrame:
        """Return ``obj`` unchanged, or raise if it reaches the hold-out while locked."""
        if self.is_open or len(obj.index) == 0:
            return obj
        last = pd.Timestamp(pd.DatetimeIndex(obj.index).max())
        if last >= HOLDOUT_START:
            raise HoldoutLocked(
                f"{what} reaches {last.date()}; the {HOLDOUT_START.date()}-"
                f"{HOLDOUT_END.date()} hold-out is locked (open it with a new, "
                "signed, dated registration)")
        return obj

    @staticmethod
    def training(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """The rows before the hold-out start."""
        return obj.loc[pd.DatetimeIndex(obj.index) < HOLDOUT_START]

    @classmethod
    def open_with(cls, path: Path, repo: Path = REPO, ledger: Path = LEDGER,
                  now: dt.datetime | None = None) -> HoldoutLock:
        """Open the lock for one registration; see the module docstring."""
        path = Path(path).resolve()
        prereg_dir = (repo / "research" / "prereg").resolve()
        if path.parent != prereg_dir:
            raise HoldoutLocked(f"{path} is not in {prereg_dir}")
        if not _DATED.match(path.name):
            raise HoldoutLocked(f"{path.name}: the registration must be new and dated "
                                "(YYYY-MM-DD-<name>.md)")
        rel = path.relative_to(repo.resolve()).as_posix()
        if rel in prereg.WITHDRAWN:
            raise HoldoutLocked(f"{rel} is withdrawn ({prereg.WITHDRAWN[rel]})")
        reg = prereg.load(path)
        if not reg.signed:
            raise HoldoutLocked(f"{path.name} is not signed with its current hash")
        body = path.read_text(encoding="utf-8")
        if str(HOLDOUT_START.date()) not in body or str(HOLDOUT_END.date()) not in body:
            raise HoldoutLocked(f"{path.name} does not name the hold-out window "
                                f"{HOLDOUT_START.date()} to {HOLDOUT_END.date()}")
        seen = []
        if ledger.exists():
            seen = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines() if x]
        if any(r["sha256"] == reg.sha256 for r in seen):
            raise HoldoutLocked(f"{path.name} has already opened the hold-out once")
        stamp = (now or dt.datetime.now(dt.UTC)).isoformat(timespec="seconds")
        with ledger.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"registration": rel, "sha256": reg.sha256,
                                 "opened_at": stamp}) + "\n")
        return cls(registration=reg.sha256)
