"""System design, case, mock rounds and study paths: keys against independent checks.

Each graded answer in a key is recomputed by the pack's ``verify.py`` by a
different route and compared here. The structural tests hold the plan's
acceptance wording to something checkable: eight design cases each naming
the code of a W25-W30 component, six timed mock rounds whose keys live only
under ``mock-rounds/keys/``, and study paths that cover every role in
section 7 of the plan with links that resolve.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEY_LINE = re.compile(r"^- \*\*([A-Z][0-9.]+)\*\* `([^`]+)`", re.M)
ID_LINE = re.compile(r"^- \*\*([A-Z][0-9.]+)\*\*", re.M)


def _load(pack: str) -> ModuleType:
    name = f"cp_verify_final_{pack.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, ROOT / pack / "verify.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PACKS = ("system-design", "case", "mock-rounds")
VERIFY = {p: _load(p) for p in PACKS}


def _keys(pack: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted((ROOT / pack / "keys").glob("*.md")):
        out.update(KEY_LINE.findall(f.read_text("utf-8")))
    return out


def _question_files(pack: str) -> list[Path]:
    if pack == "mock-rounds":
        return sorted((ROOT / pack).glob("round-*.md"))
    return [ROOT / pack / "questions.md"]


def _graded_ids(pack: str) -> list[str]:
    ids: list[str] = []
    for f in _question_files(pack):
        text = f.read_text("utf-8")
        graded = text.split("## Discussion")[0].split("## Behavioural")[0]
        ids += ID_LINE.findall(graded)
    return ids


@pytest.mark.parametrize("pack", PACKS)
def test_every_graded_question_has_one_key_and_one_check(pack: str) -> None:
    ids = _graded_ids(pack)
    assert len(ids) == len(set(ids))
    assert set(ids) == set(_keys(pack)) == set(VERIFY[pack].EXACT), pack


@pytest.mark.parametrize("pack", PACKS)
def test_question_files_hold_no_answers(pack: str) -> None:
    keys = _keys(pack)
    for f in _question_files(pack):
        for line in f.read_text("utf-8").splitlines():
            m = ID_LINE.match(line)
            if m and m.group(1) in keys:
                ans = keys[m.group(1)]
                assert f"`{ans}`" not in line
                assert len(ans) <= 3 or ans not in line, (m.group(1), ans)


CASES = [(p, pid) for p in PACKS for pid in sorted(VERIFY[p].EXACT)]


@pytest.mark.parametrize(("pack", "pid"), CASES)
def test_key_matches_independent_computation(pack: str, pid: str) -> None:
    mod = VERIFY[pack]
    written = Fraction(_keys(pack)[pid])
    computed = mod.EXACT[pid]()
    if isinstance(computed, Fraction):
        assert written == computed, (pid, written, computed)
    else:
        rel = getattr(mod, "TOL", {}).get(pid, 1e-9)
        assert float(written) == pytest.approx(computed, rel=rel, abs=1e-12), pid


SECOND = [(p, pid) for p in PACKS for pid in sorted(getattr(VERIFY[p], "SECOND_ROUTE", {}))]


@pytest.mark.parametrize(("pack", "pid"), SECOND)
def test_key_matches_second_route(pack: str, pid: str) -> None:
    got = VERIFY[pack].SECOND_ROUTE[pid]()
    assert float(Fraction(_keys(pack)[pid])) == pytest.approx(got, rel=1e-9), pid


MC = [(p, pid) for p in PACKS for pid in sorted(getattr(VERIFY[p], "MONTE_CARLO", {}))]


@pytest.mark.parametrize(("pack", "pid"), MC)
def test_key_matches_seeded_simulation(pack: str, pid: str) -> None:
    estimator, tol = VERIFY[pack].MONTE_CARLO[pid]
    assert abs(estimator() - float(Fraction(_keys(pack)[pid]))) < tol, pid


def test_a_wrong_key_is_caught() -> None:
    """Not vacuous: a key one part in 10^6 off fails, and so does an off-by-one integer."""
    sd = VERIFY["system-design"]
    assert Fraction(2049) != sd.EXACT["X08"]()
    case = VERIFY["case"]
    assert 75022.4646909 * (1 + 1e-6) != pytest.approx(case.EXACT["C06"](), rel=1e-9)


# ---------------------------------------------------------------- structure
PLAN_CODE_PATHS = (
    "qp_exec/lob/",
    "cpp/exec-core/",
    "qp_exec/mm/",
    "qp_exec/crb/",
    "qp_exec/rfq/",
    "qp_exec/sor/",
    "python/oms/",
    "python/posttrade/",
    "cpp/ull/",
)


def test_eight_design_cases_each_name_component_code() -> None:
    text = (ROOT / "system-design" / "questions.md").read_text("utf-8")
    blocks = re.split(r"^- \*\*X\d\d\*\*", text, flags=re.M)[1:]
    assert len(blocks) == 8
    for block in blocks:
        code = block.split("\n")[0]
        assert code.strip().startswith("Code:"), code
        assert any(p in code for p in PLAN_CODE_PATHS), code
        assert "*Sizing:*" in block


def test_design_cases_name_code_that_exists_where_it_has_landed() -> None:
    """`python/oms/` is on this branch; its named sub-packages must exist."""
    repo = ROOT.parent
    text = (ROOT / "system-design" / "questions.md").read_text("utf-8")
    for path in re.findall(r"`(python/oms/[\w/]*)`", text):
        assert (repo / path).exists(), path


def test_six_timed_mock_rounds_with_keys_only_under_keys() -> None:
    rounds = sorted((ROOT / "mock-rounds").glob("round-*.md"))
    keys = sorted((ROOT / "mock-rounds" / "keys").glob("round-*.md"))
    assert [r.name for r in rounds] == [k.name for k in keys] == [f"round-{i}.md" for i in range(1, 7)]
    for r in rounds:
        head = r.read_text("utf-8").splitlines()[0]
        assert re.search(r"\(\d+ minutes\)$", head), head
        assert not KEY_LINE.search(r.read_text("utf-8")), r.name
    readme = (ROOT / "mock-rounds" / "README.md").read_text("utf-8")
    assert "## Debrief" in readme
    for i in range(1, 7):
        assert f"(keys/round-{i}.md)" in readme


def test_every_discussion_question_has_a_rubric() -> None:
    for i in range(1, 7):
        q = (ROOT / "mock-rounds" / f"round-{i}.md").read_text("utf-8").split("## Discussion")[1]
        k = (
            (ROOT / "mock-rounds" / "keys" / f"round-{i}.md")
            .read_text("utf-8")
            .split("## Discussion rubric")[1]
        )
        assert ID_LINE.findall(q) == ID_LINE.findall(k) and len(ID_LINE.findall(q)) == 2


ROLES = (
    "Quant analyst / researcher",
    "Quant developer",
    "Algo developer",
    "Strats",
    "Low-latency developer",
    "Pricing quant",
    "Risk quant",
    "Hedging",
    "Execution",
    "Market data",
    "Order management (FIX/OMS)",
    "Post-trade and surveillance",
    "Central risk book and market making",
)


def test_paths_cover_every_plan_role_and_three_lengths() -> None:
    text = (ROOT / "paths" / "README.md").read_text("utf-8")
    for role in ROLES:
        assert f"| {role} |" in text, role
    for weeks in ("Four weeks", "Eight weeks", "Twelve weeks"):
        assert f"## {weeks}" in text


@pytest.mark.parametrize(
    "doc",
    ["paths/README.md", "mock-rounds/README.md", "system-design/questions.md", "case/questions.md"]
    + [f"mock-rounds/round-{i}.md" for i in range(1, 7)]
    + [f"mock-rounds/keys/round-{i}.md" for i in range(1, 7)]
    + ["system-design/keys/answers.md", "case/keys/answers.md"],
)
def test_relative_links_resolve(doc: str) -> None:
    path = ROOT / doc
    for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", path.read_text("utf-8")):
        if target.startswith("http"):
            continue
        assert (path.parent / target).resolve().exists(), (doc, target)
