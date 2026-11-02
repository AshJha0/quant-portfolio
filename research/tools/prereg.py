"""Pre-registration hashing and the result-file check.

A pre-registration is a markdown file under ``research/prereg/``. Its
registration hash is the SHA-256 of every byte before the ``## Signature``
heading, with line endings normalised to LF so a Windows checkout hashes the
same as CI. Signing appends three lines under that heading and so never
changes the hash:

    signed_by: <name>
    signed_on: <YYYY-MM-DD>
    sha256: <registration hash>

A result file is JSON with at least ``prereg`` (repo-relative path),
``prereg_sha256`` and ``period`` (``{"start": ..., "end": ...}``). The check
refuses a result when

- the hash in the result does not match the pre-registration as committed;
- the result's period reaches the hold-out (on or after ``HOLDOUT_START``)
  and the pre-registration is not signed with that same hash;
- it is a hold-out result (``"mode": "holdout"`` or a ``*_holdout*.json``
  name) and its ``provenance`` block is missing a field, records a dirty
  tree, lists no input-file hash, disagrees with ``prereg_sha256``, or is a
  dry run (dry runs never belong in ``research/results``).

Usage: ``python research/tools/prereg.py check [results_dir]`` exits 1 and
prints one line per refused file. ``prereg.py sign FILE NAME [YYYY-MM-DD]``
fills the signature block and refuses unless the registration names a
recorded simulated power result (``research/power/*.json``; ``power_errors``).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOLDOUT_START = date(2022, 1, 1)
HOLDOUT_END = date(2025, 12, 31)
SIG_HEADING = "## Signature"
_FIELD = re.compile(r"^(signed_by|signed_on|sha256):[ \t]*(.*?)[ \t]*$", re.M)


@dataclass(frozen=True)
class Registration:
    path: Path
    sha256: str
    signed_by: str
    signed_on: str
    signed_sha256: str

    @property
    def signed(self) -> bool:
        return bool(self.signed_by and self.signed_on) and self.signed_sha256 == self.sha256


def _split(text: str) -> tuple[str, str]:
    text = text.replace("\r\n", "\n")
    i = text.find("\n" + SIG_HEADING + "\n")
    if i < 0:
        raise ValueError(f"no '{SIG_HEADING}' heading")
    return text[: i + 1], text[i + 1 :]


def registration_hash(text: str) -> str:
    body, _ = _split(text)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load(path: Path) -> Registration:
    text = path.read_text(encoding="utf-8")
    _, sig = _split(text)
    fields = dict(_FIELD.findall(sig))
    return Registration(
        path=path,
        sha256=registration_hash(text),
        signed_by=fields.get("signed_by", ""),
        signed_on=fields.get("signed_on", ""),
        signed_sha256=fields.get("sha256", ""),
    )


_POWER_REF = re.compile(r"research/power/[\w.-]+\.json")


class SigningRefused(RuntimeError):
    """A registration may not be signed yet."""


def power_errors(text: str, repo: Path = REPO) -> list[str]:
    """Why the registration lacks a recorded simulated power result.

    The text must name at least one ``research/power/*.json`` file; each must
    exist and hold a non-empty ``cells`` list whose every cell has ``reps`` > 0
    and a ``rejection_rate`` (at the top of the cell or one level down): the
    rejection rate of the decision rule on simulated ground-truth data."""
    refs = sorted(set(_POWER_REF.findall(_split(text)[0])))
    if not refs:
        return ["no simulated power result (research/power/*.json) is named in the registration"]
    errs = []
    for ref in refs:
        try:
            cells = json.loads((repo / ref).read_text(encoding="utf-8")).get("cells")
        except (OSError, ValueError) as e:
            errs.append(f"{ref}: unreadable ({e})")
            continue
        if not isinstance(cells, list) or not cells:
            errs.append(f"{ref}: no 'cells'")
            continue
        for i, c in enumerate(cells):
            rates = [c.get("rejection_rate")] + [v.get("rejection_rate") for v in c.values() if isinstance(v, dict)]
            reps_ok = isinstance(c.get("reps"), int) and c["reps"] > 0
            if not reps_ok or not any(isinstance(r, (int, float)) for r in rates):
                errs.append(f"{ref}: cell {i} has no reps or rejection_rate")
    return errs


def sign(path: Path, by: str, on: str, repo: Path = REPO) -> str:
    """Fill the signature block; refuses without a recorded power result."""
    text = path.read_text(encoding="utf-8")
    if errs := power_errors(text, repo):
        raise SigningRefused("; ".join(errs))
    if load(path).signed:
        raise SigningRefused("already signed")
    date.fromisoformat(on)
    body, _ = _split(text)
    h = registration_hash(text)
    path.write_text(f"{body}{SIG_HEADING}\n\nsigned_by: {by}\nsigned_on: {on}\nsha256: {h}\n",
                    encoding="utf-8", newline="\n")
    return h


def touches_holdout(period: dict) -> bool:
    return date.fromisoformat(str(period["end"])[:10]) >= HOLDOUT_START


PROVENANCE_FIELDS = ("git_commit", "git_dirty", "input_sha256", "prereg_sha256",
                     "timestamp_utc", "packages", "dry_run")


def check_holdout_provenance(res: dict, claimed: str) -> list[str]:
    prov = res.get("provenance")
    if not isinstance(prov, dict):
        return ["hold-out result has no provenance block"]
    missing = [f for f in PROVENANCE_FIELDS if f not in prov]
    if missing:
        return [f"hold-out provenance missing {', '.join(missing)}"]
    errors = []
    if prov["git_dirty"] is not False:
        errors.append("hold-out result was produced on a dirty git tree")
    if prov["dry_run"] is not False:
        errors.append("dry-run result in the results directory")
    if not prov["input_sha256"]:
        errors.append("hold-out provenance lists no input-file hash")
    if prov["prereg_sha256"].get(res["prereg"]) != claimed:
        errors.append("provenance prereg hash disagrees with prereg_sha256")
    return errors


def check_result(result_path: Path, repo: Path = REPO) -> list[str]:
    """Return the reasons to refuse ``result_path`` (empty list = accepted)."""
    try:
        res = json.loads(result_path.read_text(encoding="utf-8"))
        reg_path = repo / res["prereg"]
        claimed = res["prereg_sha256"]
        period = res["period"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"unreadable result or missing field: {exc!r}"]
    if not reg_path.is_file():
        return [f"pre-registration not found: {res['prereg']}"]
    try:
        reg = load(reg_path)
    except ValueError as exc:
        return [f"bad pre-registration: {exc}"]
    errors = []
    if claimed != reg.sha256:
        errors.append(f"registration hash mismatch: result {claimed[:12]}, file {reg.sha256[:12]}")
    try:
        holdout = touches_holdout(period)
    except (KeyError, ValueError) as exc:
        return [*errors, f"bad period: {exc!r}"]
    if holdout and not reg.signed:
        errors.append("period reaches the 2022-2025 hold-out but the registration is not signed")
    if res.get("mode") == "holdout" or "_holdout" in result_path.name:
        errors += check_holdout_provenance(res, claimed)
    return errors


def check_dir(results_dir: Path, repo: Path = REPO) -> dict[str, list[str]]:
    return {
        str(p.relative_to(repo) if p.is_relative_to(repo) else p): errs
        for p in sorted(results_dir.rglob("*.json"))
        if (errs := check_result(p, repo))
    }


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "hash":
        for name in argv[2:]:
            reg = load(Path(name))
            print(f"{reg.sha256}  signed={reg.signed}  {name}")
        return 0
    if len(argv) >= 4 and argv[1] == "sign":
        on = argv[4] if len(argv) > 4 else date.today().isoformat()
        try:
            print(f"signed {argv[2]} sha256={sign(Path(argv[2]), argv[3], on)}")
        except SigningRefused as refusal:
            print(f"REFUSED to sign {argv[2]}: {refusal}")
            return 1
        return 0
    if len(argv) >= 2 and argv[1] == "check":
        results_dir = Path(argv[2]) if len(argv) > 2 else REPO / "research" / "results"
        if not results_dir.exists():
            print(f"no results directory {results_dir}; nothing to check")
            return 0
        bad = check_dir(results_dir)
        for name, errs in bad.items():
            for e in errs:
                print(f"REFUSED {name}: {e}")
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
