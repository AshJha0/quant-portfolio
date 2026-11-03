"""Emit the CI matrix of Python projects and the extra each one installs.

For every ``python/<area>/<NN-name>/pyproject.toml`` and for the shared
package ``python/core`` the extra is ``test`` if the project declares it,
else ``dev``, else none. The result is written as JSON to stdout and, inside
GitHub Actions, to ``$GITHUB_OUTPUT`` as ``projects=<json>`` (the 23
projects and the shared stacks) and ``core=<json>`` (``python/core``, tested by its own job and
installed before every project).

Run locally from the repository root::

    python .github/scripts/discover_projects.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def pick_extra(pyproject: Path) -> str:
    with pyproject.open("rb") as fh:
        extras = tomllib.load(fh).get("project", {}).get("optional-dependencies", {})
    for name in ("test", "dev"):
        if name in extras:
            return name
    return ""


#: Shared stacks other projects may depend on; each is also a CI project.
SHARED = ("python/backtest", "python/oms", "python/execution", "python/posttrade")

_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _members(repo: Path) -> dict[str, str]:
    """Distribution name -> repo path for every workspace package but qp-core."""
    out = {}
    for rel in ("python/data", *SHARED):
        toml = repo / rel / "pyproject.toml"
        if toml.is_file():
            with toml.open("rb") as fh:
                out[tomllib.load(fh)["project"]["name"]] = rel
    return out


def workspace_deps(repo: Path, toml: Path, extra: str) -> list[str]:
    """Paths of the workspace packages a project needs (transitively), in
    install order, from its dependencies and the chosen extra. python/core
    is installed in every job and is not listed."""
    members = _members(repo)
    seen: list[str] = []

    def visit(t: Path, ex: str) -> None:
        with t.open("rb") as fh:
            proj = tomllib.load(fh).get("project", {})
        reqs = list(proj.get("dependencies", []))
        if ex:
            reqs += proj.get("optional-dependencies", {}).get(ex, [])
        for r in reqs:
            m = _NAME.match(r)
            name = m.group(1).lower().replace("_", "-") if m else ""
            rel = members.get(name)
            if rel and rel not in seen and repo / rel != t.parent:
                visit(repo / rel / "pyproject.toml", "")
                if rel not in seen:
                    seen.append(rel)

    visit(toml, extra)
    return seen


def _entry(repo: Path, toml: Path) -> dict[str, str]:
    rel = toml.parent.relative_to(repo).as_posix()
    extra = pick_extra(toml)
    return {
        "path": rel,
        "slug": rel.removeprefix("python/").replace("/", "__"),
        "extra": extra,
        "spec": f".[{extra}]" if extra else ".",
        "deps": " ".join(workspace_deps(repo, toml, extra)),
    }


def discover(repo: Path = REPO) -> list[dict[str, str]]:
    """The 23 projects under python/<area>/NN-name and the shared stacks in
    ``SHARED`` that exist (python/core excluded)."""
    tomls = sorted(repo.glob("python/*/[0-9][0-9]-*/pyproject.toml"))
    tomls += [repo / s / "pyproject.toml" for s in SHARED if (repo / s / "pyproject.toml").is_file()]
    return [_entry(repo, t) for t in tomls]


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
