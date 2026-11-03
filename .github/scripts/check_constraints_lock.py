"""Check that requirements/ci-constraints.txt pins the same versions as uv.lock.

CI installs third-party packages from uv.lock with hashes checked
(.github/scripts/hashed_install.sh). requirements/ci-constraints.txt is kept
for local ``pip install -c`` use. This check exports the full locked set
(every member, every extra except ``live``, every root dependency group)
and fails if a package and marker pair has a different version in the two
files, or appears in one and not the other.

Run from the repository root (needs uv on PATH)::

    python .github/scripts/check_constraints_lock.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)\s*(?:;([^\\#]*))?")
CONSTRAINTS = Path("requirements/ci-constraints.txt")


def pins(text: str) -> dict[tuple[str, str], str]:
    """Map (normalised name, normalised marker) to the pinned version."""
    out: dict[tuple[str, str], str] = {}
    for line in text.splitlines():
        m = PIN.match(line.strip("\r"))
        if m:
            name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
            marker = (m.group(3) or "").strip().replace('"', "'")
            out[(name, marker)] = m.group(2)
    return out


def main() -> int:
    exported = subprocess.run(
        ["uv", "export", "--frozen", "--format", "requirements-txt", "--no-header",
         "--no-hashes", "--no-emit-workspace", "--all-packages", "--all-extras",
         "--no-extra", "live", "--all-groups"],
        check=True, capture_output=True, text=True,
    ).stdout
    lock = pins(exported)
    cons = pins(CONSTRAINTS.read_text(encoding="utf-8"))
    problems = []
    for key in sorted(lock.keys() | cons.keys()):
        a, b = cons.get(key), lock.get(key)
        if a != b:
            label = key[0] + (f" ; {key[1]}" if key[1] else "")
            problems.append(f"  {label}: ci-constraints.txt {a or '-'}, uv.lock {b or '-'}")
    if problems:
        print(f"{CONSTRAINTS} and uv.lock disagree ({len(problems)}):")
        print("\n".join(problems))
        print("Regenerate both; see requirements/README.md.")
        return 1
    print(f"{CONSTRAINTS} and uv.lock agree on {len(lock)} pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
