"""Lint the exercise room in docs/LEARN.md.

Checks, per question:

1. It has a ``<sub>Tags: ...</sub>`` line, a level tag (``L0``..``L4``) and a
   folded ``<details>`` answer.
2. Numeric answers cite evidence: a question tagged ``numeric`` must name at
   least one file (``something.py``, ``VALIDATION.md``, ...) or a test
   (``test_something``) or a module (```eq_var.evt```) that resolves to a
   file under ``python/``, in its question or answer.
3. Every cited repository path that contains a ``/`` exists.

It also reports prose lines longer than ``MAX_LINE`` characters (tables are
exempt, since a Markdown table row cannot wrap).

Run from anywhere: ``python docs/learn/tools/lint_learn.py``. Exit status 0
means clean; 1 means at least one finding, each printed as ``line: message``.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEARN = REPO / "docs" / "LEARN.md"
MAX_LINE = 400

_EXT = r"(?:py|md|rs|cpp|hpp|h|java|ipynb|json|csv|toml|ya?ml|txt|cfg|ini|html|sh|cmd)"
CITATION = re.compile(rf"(?<![\w-])(?:[\w.{{}},*-]+/)*[\w{{}},*-]+\.{_EXT}\b|\btest_\w+")
PATH = re.compile(rf"(?<![\w./-])((?:[\w.-]+/)+[\w.-]+\.{_EXT})\b")
QUESTION = re.compile(r"^\*\*Q(\d+)\. ")
SET = re.compile(r"^## Set (\d+):")
TAGS = re.compile(r"^<sub>Tags: (.*)</sub>\s*$")
LEVEL = re.compile(r"`L[0-4]`")
MODULE = re.compile(r"`([a-z][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+)`")


@dataclass
class Question:
    set_no: int
    number: int
    line: int
    text: list[str]

    @property
    def body(self) -> str:
        return "\n".join(self.text)


def split_questions(lines: list[str]) -> list[Question]:
    """Cut the exercise room into questions (a question runs to the next one or next set)."""
    out: list[Question] = []
    set_no = 0
    cur: Question | None = None
    for i, line in enumerate(lines, start=1):
        if (m := SET.match(line)) is not None:
            set_no = int(m.group(1))
            cur = None
            continue
        if line.startswith("## ") or line.startswith("### Primer"):
            cur = None
            continue
        if (m := QUESTION.match(line)) is not None and set_no:
            cur = Question(set_no, int(m.group(1)), i, [line])
            out.append(cur)
            continue
        if cur is not None:
            cur.text.append(line)
    return out


def _path_exists(p: str) -> bool:
    if any(c in p for c in "{}*,"):
        return True  # a pattern, not one file
    for base in (REPO, REPO / "docs"):
        if (base / p).exists():
            return True
    # Paths quoted relative to a package (src/eq_x/y.py): accept if some file ends with it.
    return any(REPO.glob(f"**/{p}"))


_MODULE_CACHE: dict[str, bool] = {}


def module_exists(dotted: str) -> bool:
    """True when some prefix of ``a.b.c`` names a Python module or package in the repo."""
    if dotted not in _MODULE_CACHE:
        parts = dotted.split(".")
        found = False
        for k in range(len(parts), 1, -1):
            rel = "/".join(parts[:k])
            if any(REPO.glob(f"python/**/{rel}.py")) or any(REPO.glob(f"python/**/{rel}/__init__.py")):
                found = True
                break
        _MODULE_CACHE[dotted] = found
    return _MODULE_CACHE[dotted]


def cites_evidence(body: str) -> bool:
    """A file, a test, or a module (``pkg.module``) that resolves to a file in the repo."""
    if CITATION.search(body):
        return True
    return any(module_exists(m) for m in MODULE.findall(body))


def lint(text: str) -> list[str]:
    lines = text.splitlines()
    findings: list[str] = []
    in_code = False
    for i, line in enumerate(lines, start=1):
        if line.startswith("```"):
            in_code = not in_code
        if len(line) > MAX_LINE and not in_code and not line.lstrip().startswith("|"):
            findings.append(f"{i}: line is {len(line)} characters (max {MAX_LINE})")
    checked: dict[str, bool] = {}
    for q in split_questions(lines):
        where = f"{q.line}: Set {q.set_no} Q{q.number}"
        tag_line = next((t for t in q.text if TAGS.match(t)), None)
        if tag_line is None:
            findings.append(f"{where}: no Tags line")
            continue
        if not LEVEL.search(tag_line):
            findings.append(f"{where}: no level tag (L0..L4)")
        if "<details>" not in q.body:
            findings.append(f"{where}: answer not folded in <details>")
        body_wo_tags = q.body.replace(tag_line, "")
        if "`numeric`" in tag_line and not cites_evidence(body_wo_tags):
            findings.append(f"{where}: numeric answer cites no file or test")
        for p in PATH.findall(body_wo_tags):
            if p not in checked:
                checked[p] = _path_exists(p)
            if not checked[p]:
                findings.append(f"{where}: cited path does not exist: {p}")
    return findings


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else LEARN
    findings = lint(path.read_text(encoding="utf-8"))
    for f in findings:
        print(f)
    print(f"{len(findings)} finding(s) in {path.name}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
