"""Model inventory: the ``@model`` tag and the inventory check (W11).

The plain idea
--------------
A bank keeps a list of every model it uses: what it is for, who owns it,
how much damage a wrong answer would do (its tier), what it assumes, where
it breaks, and when someone last checked it. Supervisors require this
(Federal Reserve SR 11-7, 2011; PRA SS1/23, 2023). This repository keeps
the same list in ``docs/models/models.toml`` and ties it to the code with a
decorator, so the list cannot quietly drift from what the code contains.

The formal statement
--------------------
*Model*: any function or class whose output is a price, a sensitivity, a
risk measure or a forecast. Each one is tagged in the source with
``@model(id="...")``. The check :func:`check_inventory` then requires:

1. every id used by a decorator has exactly one inventory entry;
2. every inventory entry names entrypoints (``path::qualname``) that exist,
   and when ``decorated = true`` (the default) each entrypoint carries the
   decorator with the entry's id; ``decorated = false`` marks an entry whose
   decorator has not been applied yet, and then the entrypoint must not
   carry one, so the flag cannot go stale in either direction;
3. every test the entry cites (``path::test_name``) exists;
4. dates are ISO dates, ``validated_on <= next_review``, ``validated_on``
   is not in the future, and the review interval is at most 6 months for
   tier 1 and 12 months otherwise;
5. a tier 1 entry cites at least one independent check (another library,
   a closed form computed independently, or another language's engine).

The decorator returns the object unchanged apart from one attribute, so a
tagged function keeps its signature, its speed and its docstring.

Worked example
--------------
>>> @model(id="DOC-EXAMPLE")
... def forward(spot: float, carry: float, t: float) -> float:
...     return spot * (1.0 + carry * t)
>>> forward(100.0, 0.05, 1.0)
105.0
>>> model_id(forward)
'DOC-EXAMPLE'
>>> scan_source('@model(id="EQ-X")\\ndef f(): ...\\n', "m.py")
[DecoratorSite(model_id='EQ-X', path='m.py', qualname='f', line=2)]

Run the check from the repository root::

    python -m qp_core.model check docs/models/models.toml
"""

from __future__ import annotations

import argparse
import ast
import datetime as _dt
import re
import sys
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

__all__ = [
    "ID_PATTERN",
    "KINDS",
    "STATUSES",
    "DecoratorSite",
    "ModelEntry",
    "check_inventory",
    "load_inventory",
    "model",
    "model_id",
    "registered",
    "scan_source",
    "scan_tree",
]

#: Upper-case words joined by hyphens, at least two words: ``EQ-OPT-BS``.
ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+$")
KINDS = frozenset({"price", "sensitivity", "risk", "forecast"})
#: ``independently-checked``: agrees with an independent implementation or
#: closed form in a test. ``self-tested``: tested only against its own
#: properties (parity, limits, round trips). ``not-validated``: neither.
STATUSES = frozenset({"independently-checked", "self-tested", "not-validated", "retired"})
_ATTR = "__qp_model_id__"
_MAX_REVIEW_DAYS = {1: 184, 2: 366, 3: 366}

_F = TypeVar("_F")
_REGISTRY: dict[str, list[str]] = {}


def model(*, id: str) -> Callable[[_F], _F]:
    """Tag a function or class as model ``id``; returns it unchanged.

    Raises ``ValueError`` for an id that does not match :data:`ID_PATTERN`.
    """
    if not isinstance(id, str) or not ID_PATTERN.match(id):
        raise ValueError(f"model id {id!r} must match {ID_PATTERN.pattern}")

    def tag(obj: _F) -> _F:
        setattr(obj, _ATTR, id)
        where = f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', '?')}"
        _REGISTRY.setdefault(id, [])
        if where not in _REGISTRY[id]:
            _REGISTRY[id].append(where)
        return obj

    return tag


def model_id(obj: object) -> str | None:
    """The id an object was tagged with, or ``None``."""
    value = getattr(obj, _ATTR, None)
    return value if isinstance(value, str) else None


def registered() -> dict[str, list[str]]:
    """Ids tagged so far in this process (imported modules only)."""
    return {k: list(v) for k, v in _REGISTRY.items()}


# --------------------------------------------------------------------------
# Static scan: find decorators without importing anything.


@dataclass(frozen=True)
class DecoratorSite:
    model_id: str
    path: str
    qualname: str
    line: int


def _decorator_id(dec: ast.expr) -> str | None:
    if not isinstance(dec, ast.Call):
        return None
    name = dec.func.attr if isinstance(dec.func, ast.Attribute) else getattr(dec.func, "id", None)
    if name != "model":
        return None
    for kw in dec.keywords:
        if kw.arg == "id" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _definitions(
    tree: ast.Module,
) -> Iterable[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]]:
    stack: list[tuple[str, ast.AST]] = [("", tree)]
    while stack:
        prefix, node = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                qual = f"{prefix}{child.name}"
                yield qual, child
                stack.append((qual + ".", child))


def scan_source(source: str, path: str) -> list[DecoratorSite]:
    """Every ``@model(id=...)`` site in one module's source."""
    out = []
    for qual, node in _definitions(ast.parse(source, filename=path)):
        for dec in node.decorator_list:
            mid = _decorator_id(dec)
            if mid is not None:
                out.append(DecoratorSite(mid, path, qual, node.lineno))
    return sorted(out, key=lambda s: (s.path, s.line))


def _definition_index(source: str, path: str) -> dict[str, list[str]]:
    """qualname -> list of decorator ids on it (empty when untagged)."""
    index: dict[str, list[str]] = {}
    for qual, node in _definitions(ast.parse(source, filename=path)):
        ids = [i for i in (_decorator_id(d) for d in node.decorator_list) if i is not None]
        index.setdefault(qual, []).extend(ids)
    return index


def scan_tree(repo_root: Path, roots: Sequence[str]) -> list[DecoratorSite]:
    """Every decorator site under ``roots`` (paths relative to the repo)."""
    sites: list[DecoratorSite] = []
    for root in roots:
        for f in sorted((repo_root / root).rglob("*.py")):
            if any(part in {".venv", "build", "node_modules"} for part in f.parts):
                continue
            rel = f.relative_to(repo_root).as_posix()
            sites.extend(scan_source(f.read_text(encoding="utf-8"), rel))
    return sites


# --------------------------------------------------------------------------
# The inventory file.


@dataclass(frozen=True)
class ModelEntry:
    id: str
    name: str
    owner: str
    tier: int
    kind: str
    purpose: str
    entrypoints: tuple[str, ...]
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]
    validation_status: str
    validated_on: _dt.date
    next_review: _dt.date
    kpis: tuple[str, ...]
    tests: tuple[str, ...]
    independent_checks: tuple[str, ...] = ()
    decorated: bool = True
    notes: str = ""
    problems: tuple[str, ...] = field(default=(), compare=False)


_REQUIRED = (
    "id",
    "name",
    "owner",
    "tier",
    "kind",
    "purpose",
    "entrypoints",
    "assumptions",
    "limitations",
    "validation_status",
    "validated_on",
    "next_review",
    "kpis",
    "tests",
)


def _date(value: Any) -> _dt.date | None:
    if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return _dt.date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _strings(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, list) and all(isinstance(v, str) and v.strip() for v in value):
        return tuple(value)
    return None


def _entry(raw: dict[str, Any]) -> ModelEntry:
    problems: list[str] = []
    rid = raw.get("id", "?")
    for key in _REQUIRED:
        if key not in raw:
            problems.append(f"{rid}: missing field {key!r}")
    if isinstance(rid, str) and not ID_PATTERN.match(rid):
        problems.append(f"{rid}: id does not match {ID_PATTERN.pattern}")
    tier = raw.get("tier")
    if tier not in (1, 2, 3):
        problems.append(f"{rid}: tier must be 1, 2 or 3, got {tier!r}")
    if raw.get("kind") not in KINDS:
        problems.append(f"{rid}: kind must be one of {sorted(KINDS)}, got {raw.get('kind')!r}")
    if raw.get("validation_status") not in STATUSES:
        problems.append(f"{rid}: validation_status must be one of {sorted(STATUSES)}")
    lists: dict[str, tuple[str, ...]] = {}
    for key in ("entrypoints", "assumptions", "limitations", "kpis", "tests", "independent_checks"):
        if key not in raw and key == "independent_checks":
            lists[key] = ()
            continue
        got = _strings(raw.get(key))
        if got is None or (key != "independent_checks" and not got):
            problems.append(f"{rid}: {key} must be a non-empty list of strings")
            got = ()
        lists[key] = got
    dates = {k: _date(raw.get(k)) for k in ("validated_on", "next_review")}
    for k, v in dates.items():
        if v is None:
            problems.append(f"{rid}: {k} must be an ISO date")
    for key in ("name", "owner", "purpose"):
        if not (isinstance(raw.get(key), str) and raw[key].strip()):
            problems.append(f"{rid}: {key} must be a non-empty string")
    unknown = set(raw) - set(_REQUIRED) - {"independent_checks", "decorated", "notes"}
    if unknown:
        problems.append(f"{rid}: unknown fields {sorted(unknown)}")
    return ModelEntry(
        id=str(rid),
        name=str(raw.get("name", "")),
        owner=str(raw.get("owner", "")),
        tier=tier if isinstance(tier, int) else 0,
        kind=str(raw.get("kind", "")),
        purpose=str(raw.get("purpose", "")),
        entrypoints=lists["entrypoints"],
        assumptions=lists["assumptions"],
        limitations=lists["limitations"],
        validation_status=str(raw.get("validation_status", "")),
        validated_on=dates["validated_on"] or _dt.date.min,
        next_review=dates["next_review"] or _dt.date.min,
        kpis=lists["kpis"],
        tests=lists["tests"],
        independent_checks=lists["independent_checks"],
        decorated=bool(raw.get("decorated", True)),
        notes=str(raw.get("notes", "")),
        problems=tuple(problems),
    )


def load_inventory(path: Path) -> tuple[list[ModelEntry], list[str]]:
    """Parse the inventory TOML: ``[[model]]`` tables plus ``[inventory]``.

    Returns the entries and the field-level problems found while parsing.
    """
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_models = data.get("model", [])
    if not isinstance(raw_models, list):
        return [], [f"{path}: [[model]] must be an array of tables"]
    entries = [_entry(r) for r in raw_models]
    problems = [p for e in entries for p in e.problems]
    return entries, problems


def _split_ref(ref: str) -> tuple[str, str] | None:
    if "::" not in ref:
        return None
    path, _, name = ref.partition("::")
    return (path, name) if path and name else None


def check_inventory(
    entries: Sequence[ModelEntry],
    repo_root: Path,
    source_roots: Sequence[str],
    *,
    today: _dt.date | None = None,
) -> list[str]:
    """All inventory problems as readable strings; empty means it passes."""
    today = today or _dt.date.today()
    problems = [p for e in entries for p in e.problems]
    ids: dict[str, int] = {}
    for e in entries:
        ids[e.id] = ids.get(e.id, 0) + 1
    problems += [f"{i}: listed {n} times" for i, n in sorted(ids.items()) if n > 1]

    # 1. Every decorator id has an entry.
    for site in scan_tree(repo_root, source_roots):
        if site.model_id not in ids:
            problems.append(
                f"{site.path}:{site.line}: {site.qualname} is tagged @model(id={site.model_id!r}) "
                "but docs/models/models.toml has no entry for it"
            )

    index_cache: dict[str, dict[str, list[str]] | None] = {}

    def index(rel: str) -> dict[str, list[str]] | None:
        if rel not in index_cache:
            f = repo_root / rel
            index_cache[rel] = _definition_index(f.read_text(encoding="utf-8"), rel) if f.is_file() else None
        return index_cache[rel]

    for e in entries:
        # 2. Entrypoints exist and carry (or deliberately lack) the tag.
        for ref in e.entrypoints:
            parts = _split_ref(ref)
            if parts is None:
                problems.append(f"{e.id}: entrypoint {ref!r} is not 'path::qualname'")
                continue
            idx = index(parts[0])
            if idx is None:
                problems.append(f"{e.id}: entrypoint file {parts[0]} does not exist")
                continue
            if parts[1] not in idx:
                problems.append(f"{e.id}: {parts[1]} is not defined in {parts[0]}")
                continue
            tags = idx[parts[1]]
            if e.decorated and e.id not in tags:
                problems.append(f"{e.id}: {ref} is not tagged @model(id={e.id!r})")
            if not e.decorated and tags:
                problems.append(f"{e.id}: {ref} is tagged {tags}; set decorated = true")
            for other in tags:
                if other != e.id:
                    problems.append(f"{e.id}: {ref} is tagged with a different id {other!r}")
        # 3. Cited tests exist.
        for ref in (*e.tests, *e.independent_checks):
            parts = _split_ref(ref)
            if parts is None:
                if not (repo_root / ref).exists():
                    problems.append(f"{e.id}: cited test {ref} does not exist")
                continue
            idx = index(parts[0]) if parts[0].endswith(".py") else None
            if parts[0].endswith(".py"):
                if idx is None:
                    problems.append(f"{e.id}: test file {parts[0]} does not exist")
                elif parts[1] not in idx:
                    problems.append(f"{e.id}: test {parts[1]} not found in {parts[0]}")
            else:
                f = repo_root / parts[0]
                if not f.is_file():
                    problems.append(f"{e.id}: test file {parts[0]} does not exist")
                elif not re.search(r"\b" + re.escape(parts[1]) + r"\b", f.read_text(encoding="utf-8")):
                    problems.append(f"{e.id}: test {parts[1]} not found in {parts[0]}")
        # 4. Dates.
        if e.validated_on > today:
            problems.append(f"{e.id}: validated_on {e.validated_on} is in the future")
        if e.next_review < e.validated_on:
            problems.append(f"{e.id}: next_review {e.next_review} is before validated_on")
        limit = _MAX_REVIEW_DAYS.get(e.tier)
        if limit is not None and (e.next_review - e.validated_on).days > limit:
            problems.append(f"{e.id}: tier {e.tier} review interval exceeds {limit} days")
        # 5. Tier 1 needs an independent check.
        if e.tier == 1 and not e.independent_checks:
            problems.append(f"{e.id}: tier 1 model cites no independent check")
        if e.validation_status == "independently-checked" and not e.independent_checks:
            problems.append(f"{e.id}: status independently-checked but no independent check cited")
    return problems


#: Where model code lives, relative to the repository root.
DEFAULT_SOURCE_ROOTS = ("python", "research", "bindings")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m qp_core.model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    chk = sub.add_parser("check", help="check the inventory against the source tree")
    chk.add_argument("inventory", type=Path, nargs="?", default=Path("docs/models/models.toml"))
    chk.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    entries, _ = load_inventory(args.root / args.inventory)
    problems = check_inventory(entries, args.root, DEFAULT_SOURCE_ROOTS)
    for p in problems:
        print(p)
    pending = sum(1 for e in entries if not e.decorated)
    print(f"{len(entries)} models, {pending} awaiting their decorator, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
