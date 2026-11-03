"""W11: the model inventory matches the code.

The repository's inventory (``docs/models/models.toml``) must pass
:func:`qp_core.model.check_inventory` against the source tree. The other
tests build a small fake repository in ``tmp_path`` and show that each
kind of drift the plan names is caught: a decorated function with no
entry, an entry whose tests do not exist, and the rest of the rules.

Run from the repository root::

    python -m pytest -p no:cacheprovider -q tests/test_model_inventory.py
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Load this checkout's qp_core/model.py by path: an installed qp_core may
# come from another checkout (an editable install elsewhere, or an older
# wheel) that has no model module or an older one.
_spec = importlib.util.spec_from_file_location(
    "_qp_model_under_test", ROOT / "python" / "core" / "src" / "qp_core" / "model.py"
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

DEFAULT_SOURCE_ROOTS = _mod.DEFAULT_SOURCE_ROOTS
check_inventory = _mod.check_inventory
load_inventory = _mod.load_inventory
main = _mod.main
model = _mod.model
model_id = _mod.model_id
registered = _mod.registered
scan_source = _mod.scan_source

INVENTORY = ROOT / "docs" / "models" / "models.toml"
TODAY = dt.date(2026, 11, 3)


def test_repository_inventory_passes() -> None:
    entries, _ = load_inventory(INVENTORY)
    problems = check_inventory(entries, ROOT, DEFAULT_SOURCE_ROOTS)
    assert problems == []
    assert len(entries) >= 30


def test_every_kind_and_tier_one_model_has_an_independent_check() -> None:
    entries, _ = load_inventory(INVENTORY)
    assert {e.kind for e in entries} == {"price", "sensitivity", "risk", "forecast"}
    assert all(e.independent_checks for e in entries if e.tier == 1)


def test_every_entry_has_a_validation_section() -> None:
    entries, _ = load_inventory(INVENTORY)
    pack = (ROOT / "docs" / "models" / "VALIDATION.md").read_text(encoding="utf-8")
    missing = [e.id for e in entries if f"`{e.id}`" not in pack]
    assert missing == []


# ---------------------------------------------------------------------------
# A fake repository for the failure modes.

SRC = """
from qp_core.model import model

@model(id="EQ-TOY-PRICE")
def price(x):
    return x

def untagged(x):
    return x
"""

TESTS = """
def test_price():
    assert True

class TestGroup:
    def test_inner(self):
        assert True
"""


def _entry(**over: object) -> str:
    fields: dict[str, object] = {
        "id": '"EQ-TOY-PRICE"',
        "name": '"Toy"',
        "owner": '"maintainer"',
        "tier": "2",
        "kind": '"price"',
        "purpose": '"test"',
        "entrypoints": '["pkg/src/toy.py::price"]',
        "assumptions": '["none"]',
        "limitations": '["none"]',
        "validation_status": '"self-tested"',
        "validated_on": "2026-11-01",
        "next_review": "2027-11-01",
        "kpis": '["error"]',
        "tests": '["pkg/tests/test_toy.py::test_price", "pkg/tests/test_toy.py::TestGroup.test_inner"]',
    }
    fields.update(over)
    return "[[model]]\n" + "".join(f"{k} = {v}\n" for k, v in fields.items() if v is not None)


def _repo(tmp_path: Path, toml: str, src: str = SRC) -> list[str]:
    (tmp_path / "pkg" / "src").mkdir(parents=True)
    (tmp_path / "pkg" / "tests").mkdir()
    (tmp_path / "pkg" / "src" / "toy.py").write_text(src, encoding="utf-8")
    (tmp_path / "pkg" / "tests" / "test_toy.py").write_text(TESTS, encoding="utf-8")
    inv = tmp_path / "models.toml"
    inv.write_text(toml, encoding="utf-8")
    entries, _ = load_inventory(inv)
    problems: list[str] = check_inventory(entries, tmp_path, ["pkg"], today=TODAY)
    return problems


def test_clean_fake_repository_passes(tmp_path: Path) -> None:
    assert _repo(tmp_path, _entry()) == []


def test_decorated_function_without_entry_fails(tmp_path: Path) -> None:
    src = SRC + '\n@model(id="EQ-TOY-NEW")\ndef new(x):\n    return x\n'
    problems = _repo(tmp_path, _entry(), src)
    assert len(problems) == 1
    assert "EQ-TOY-NEW" in problems[0] and "no entry" in problems[0]


def test_missing_test_function_fails(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry(tests='["pkg/tests/test_toy.py::test_gone"]'))
    assert problems == ["EQ-TOY-PRICE: test test_gone not found in pkg/tests/test_toy.py"]


def test_missing_test_file_fails(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry(tests='["pkg/tests/test_nope.py::test_price"]'))
    assert problems == ["EQ-TOY-PRICE: test file pkg/tests/test_nope.py does not exist"]


def test_missing_entrypoint_fails(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry(entrypoints='["pkg/src/toy.py::gone"]'))
    assert problems == ["EQ-TOY-PRICE: gone is not defined in pkg/src/toy.py"]


def test_decorated_flag_must_match_source(tmp_path: Path) -> None:
    # Entry says "not decorated yet" but the source carries the tag.
    problems = _repo(tmp_path, _entry(decorated="false"))
    assert any("set decorated = true" in p for p in problems)


def test_untagged_entrypoint_fails(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry(entrypoints='["pkg/src/toy.py::price", "pkg/src/toy.py::untagged"]'))
    assert problems == ["EQ-TOY-PRICE: pkg/src/toy.py::untagged is not tagged @model(id='EQ-TOY-PRICE')"]


def test_duplicate_entries_fail(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry() + "\n" + _entry())
    assert problems == ["EQ-TOY-PRICE: listed 2 times"]


def test_tier_one_needs_independent_check_and_short_review(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry(tier="1"))
    assert "EQ-TOY-PRICE: tier 1 model cites no independent check" in problems
    assert "EQ-TOY-PRICE: tier 1 review interval exceeds 184 days" in problems


def test_dates_are_checked(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry(validated_on="2027-01-01", next_review="2026-12-01"))
    assert "EQ-TOY-PRICE: validated_on 2027-01-01 is in the future" in problems
    assert "EQ-TOY-PRICE: next_review 2026-12-01 is before validated_on" in problems


def test_bad_fields_are_reported(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry(tier="4", kind='"vibes"', kpis="[]", owner=None))
    joined = "\n".join(problems)
    assert "tier must be 1, 2 or 3" in joined
    assert "kind must be one of" in joined
    assert "kpis must be a non-empty list" in joined
    assert "missing field 'owner'" in joined


def test_independently_checked_needs_a_citation(tmp_path: Path) -> None:
    problems = _repo(tmp_path, _entry(validation_status='"independently-checked"'))
    assert problems == ["EQ-TOY-PRICE: status independently-checked but no independent check cited"]


# ---------------------------------------------------------------------------
# The decorator itself.


def test_decorator_returns_the_same_object() -> None:
    def f(x: float) -> float:
        return 2 * x

    g = model(id="TEST-SAME-OBJECT")(f)
    assert g is f and g(2.0) == 4.0
    assert model_id(f) == "TEST-SAME-OBJECT"
    assert any(w.endswith("f") for w in registered()["TEST-SAME-OBJECT"])


@pytest.mark.parametrize("bad", ["lower-case", "NOHYPHEN", "-LEAD", "TRAIL-", "A--B", ""])
def test_decorator_rejects_bad_ids(bad: str) -> None:
    with pytest.raises(ValueError, match="must match"):
        model(id=bad)


def test_scan_finds_methods_and_attribute_form() -> None:
    src = 'import qp_core.model as m\nclass K:\n    @m.model(id="EQ-K-M")\n    def meth(self): ...\n'
    sites = scan_source(src, "k.py")
    assert [(s.model_id, s.qualname, s.line) for s in sites] == [("EQ-K-M", "K.meth", 4)]


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(INVENTORY.relative_to(ROOT)), "--root", str(ROOT)]) == 0
    assert "0 problems" in capsys.readouterr().out
    bad = tmp_path / "bad.toml"
    bad.write_text(_entry(tests='["nowhere.py::test_x"]'), encoding="utf-8")
    assert main(["check", str(bad), "--root", str(tmp_path)]) == 1
