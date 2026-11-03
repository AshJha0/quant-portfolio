"""Public-API diff of every Python package against a git ref (griffe).

The public API is each package's ``__all__`` (docs/VERSIONING.md). For
python/core and every project under python/<area>/<nn>-*, this runs::

    griffe check <package> --search <project>/src --against <ref>

and also compares the package's ``__all__`` (read statically from
``__init__.py``) at the ref with the working tree. It fails if a name left
``__all__`` or griffe reports a breaking change: a removed or renamed
public name, a removed parameter, a parameter that lost its default, a
changed parameter kind. Additions pass. A package that does not exist at
the ref (new this change) is skipped and reported.

The bindings are not checked here: their names come from compiled
extensions that griffe cannot read without building them. Their API is
covered by the binding tests and the ``__init__.pyi`` stubs.

Run from the repository root (needs griffe, the ``api`` dependency group)::

    python .github/scripts/api_diff.py origin/main
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def packages() -> list[tuple[str, Path]]:
    """(import name, src dir) for python/core and every project."""
    out = []
    projects = [REPO / "python/core", *sorted(REPO.glob("python/*/[0-9][0-9]-*"))]
    for proj in projects:
        src = proj / "src"
        for pkg in sorted(p for p in src.iterdir() if (p / "__init__.py").is_file()):
            out.append((pkg.name, src.relative_to(REPO)))
    return out


def exists_at(ref: str, path: Path) -> bool:
    r = subprocess.run(["git", "cat-file", "-e", f"{ref}:{path.as_posix()}"], cwd=REPO,
                       capture_output=True)
    return r.returncode == 0


def dunder_all(source: str) -> set[str] | None:
    """Names in a module's ``__all__`` (``=`` and ``+=`` of literal lists), or None."""
    names: set[str] = set()
    found = False
    for node in ast.parse(source).body:
        target, value = None, None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            target, value = node.target, node.value
        if not (isinstance(target, ast.Name) and target.id == "__all__") or value is None:
            continue
        try:
            items = ast.literal_eval(value)
        except ValueError:
            return None  # computed __all__: griffe alone decides
        if isinstance(node, ast.Assign):
            names = set()
        names |= set(items)
        found = True
    return names if found else None


def removed_from_all(ref: str, init: Path) -> list[str]:
    old_src = subprocess.run(["git", "show", f"{ref}:{init.as_posix()}"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    old = dunder_all(old_src)
    new = dunder_all((REPO / init).read_text(encoding="utf-8"))
    if old is None or new is None:
        return []
    return sorted(old - new)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref", help="git ref to compare against, e.g. origin/main")
    args = ap.parse_args(argv)

    broken, skipped = [], []
    for name, src in packages():
        if not exists_at(args.ref, src / name / "__init__.py"):
            skipped.append(name)
            continue
        r = subprocess.run(
            [sys.executable, "-m", "griffe", "check", name, "--search", src.as_posix(),
             "--against", args.ref, "--format", "verbose"],
            cwd=REPO, capture_output=True, text=True,
        )
        report = (r.stdout + r.stderr).strip()
        gone = removed_from_all(args.ref, src / name / "__init__.py")
        if gone:
            report = f"removed from __all__: {', '.join(gone)}\n{report}".strip()
        if r.returncode != 0 or gone:
            broken.append(name)
            print(f"::group::{name}: breaking changes")
            print(report)
            print("::endgroup::")
        else:
            print(f"{name}: no breaking change")
    if skipped:
        print(f"not at {args.ref} (new, nothing to compare): {', '.join(skipped)}")
    if broken:
        names = ", ".join(broken)
        print(f"::error::public API break in {names}. If intended, label the pull request "
              "'api-break' and add the docs/MIGRATION.md entry (docs/VERSIONING.md).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
