"""Tests for the LEARN.md lint (run: python -m pytest docs/learn/tools)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lint_learn as L

GOOD = """## Set 1: Demo

**Q1. A numeric question?**

<sub>Tags: `L2`, `demo`, `numeric`</sub>

<details><summary>Answer</summary>

It is 0.25, from `python/core/src/qp_core/normal.py`.

</details>
"""


def test_the_published_learn_file_is_clean() -> None:
    assert L.lint(L.LEARN.read_text(encoding="utf-8")) == []


def test_a_clean_question_passes() -> None:
    assert L.lint(GOOD) == []


def test_numeric_answer_without_a_source_fails() -> None:
    text = GOOD.replace(", from `python/core/src/qp_core/normal.py`", "")
    assert any("cites no file or test" in f for f in L.lint(text))


def test_a_test_name_counts_as_a_source() -> None:
    text = GOOD.replace("`python/core/src/qp_core/normal.py`", "`test_norm_cdf_tail`")
    assert L.lint(text) == []


def test_a_resolvable_module_counts_as_a_source() -> None:
    text = GOOD.replace("`python/core/src/qp_core/normal.py`", "`qp_core.normal`")
    assert L.lint(text) == []


def test_an_unknown_module_is_not_a_source() -> None:
    text = GOOD.replace("`python/core/src/qp_core/normal.py`", "`qp_core.no_such_module`")
    assert any("cites no file or test" in f for f in L.lint(text))


def test_a_missing_cited_path_fails() -> None:
    text = GOOD.replace("normal.py", "abnormal.py")
    assert any("does not exist" in f for f in L.lint(text))


def test_missing_level_and_details_fail() -> None:
    text = GOOD.replace("`L2`, ", "").replace("<details><summary>Answer</summary>", "")
    found = L.lint(text)
    assert any("no level tag" in f for f in found)
    assert any("<details>" in f for f in found)


def test_long_prose_line_fails_but_table_row_does_not() -> None:
    long = "word " * 100
    assert any("characters" in f for f in L.lint(long))
    assert L.lint("| " + long + " |") == []
