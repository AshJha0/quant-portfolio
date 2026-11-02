"""Emit the CI matrix of Python projects and the extra each one installs.

For every ``python/<area>/<NN-name>/pyproject.toml`` and for the shared
package ``python/core`` the extra is ``test`` if the project declares it,
else ``dev``, else none. The result is written as JSON to stdout and, inside
GitHub Actions, to ``$GITHUB_OUTPUT`` as ``projects=<json>`` (the 23
projects) and ``core=<json>`` (``python/core``, tested by its own job and
installed before every project).

Run locally from the repository root::

    python .github/scripts/discover_projects.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[2]


def pick_extra(pyproject: Path) -> str:
    with pyproject.open("rb") as fh:
        extras = tomllib.load(fh).get("project", {}).get("optional-dependencies", {})
    for name in ("test", "dev"):
        if name in extras:
            return name
    return ""


def _entry(repo: Path, toml: Path) -> dict[str, str]:
    rel = toml.parent.relative_to(repo).as_posix()
    extra = pick_extra(toml)
    return {
        "path": rel,
        "slug": rel.removeprefix("python/").replace("/", "__"),
        "extra": extra,
        "spec": f".[{extra}]" if extra else ".",
    }


def discover(repo: Path = REPO) -> list[dict[str, str]]:
    """The 23 projects under python/<area>/NN-name (python/core excluded)."""
    return [_entry(repo, t) for t in sorted(repo.glob("python/*/[0-9][0-9]-*/pyproject.toml"))]


def discover_core(repo: Path = REPO) -> dict[str, str] | None:
    """The shared package python/core (slug ``core``), or None if absent."""
    toml = repo / "python" / "core" / "pyproject.toml"
    return _entry(repo, toml) if toml.is_file() else None


def main() -> int:
    projects = discover()
    if not projects:
        print("no Python projects found", file=sys.stderr)
        return 1
    core = discover_core()
    if core is None:
        print("python/core/pyproject.toml not found", file=sys.stderr)
        return 1
    print(json.dumps({"core": core, "projects": projects}, indent=2))
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            fh.write("projects=" + json.dumps(projects, separators=(",", ":")) + "\n")
            fh.write("core=" + json.dumps(core, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
