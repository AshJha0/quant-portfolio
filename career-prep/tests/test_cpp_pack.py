"""C++ pack: every "what does this print" answer and every exercise solution is run.

* C01-C11: each snippet in ``career-prep/cpp/snippets`` is compiled (with
  the address and undefined-behaviour sanitisers) and run; its output must
  equal the answer written in the key.
* C12-C20: the reference solutions in ``career-prep/cpp/solutions`` are
  exercised by ``checks/check_single.cpp`` (ASan + UBSan, all warnings as
  errors) and ``checks/check_threads.cpp`` (ThreadSanitizer).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cxx_build import SAN_FLAGS, compile_cpp, run

PACK = Path(__file__).resolve().parents[1] / "cpp"
SNIPPETS = PACK / "snippets"
SOLUTIONS = PACK / "solutions"
CHECKS = PACK / "checks"

ITEM = re.compile(r"^- \*\*(C\d\d)\*\*", re.M)
PRINTS = re.compile(r"^- \*\*(C\d\d)\*\* Prints `([^`]*)`", re.M)


def _key_text() -> str:
    return (PACK / "keys" / "answers.md").read_text("utf-8")


def _printed() -> dict[str, str]:
    return dict(PRINTS.findall(_key_text()))


def test_questions_and_key_cover_the_same_items() -> None:
    q = ITEM.findall((PACK / "questions.md").read_text("utf-8"))
    k = ITEM.findall(_key_text())
    assert len(q) == len(set(q)) == 24
    assert set(q) == set(k)


def test_every_snippet_has_a_printed_answer_and_vice_versa() -> None:
    assert set(_printed()) == {p.stem for p in SNIPPETS.glob("C*.cpp")}


def test_questions_file_holds_no_answers() -> None:
    text = (PACK / "questions.md").read_text("utf-8")
    for ans in _printed().values():
        assert f"`{ans}`" not in text


@pytest.mark.parametrize("cid", sorted(p.stem for p in SNIPPETS.glob("C*.cpp")))
def test_snippet_prints_what_the_key_says(cid: str) -> None:
    exe = compile_cpp([SNIPPETS / f"{cid}.cpp"], include=[SNIPPETS], strict=False)
    assert run(exe).strip() == _printed()[cid]


def test_single_threaded_solutions_under_asan_ubsan() -> None:
    exe = compile_cpp([CHECKS / "check_single.cpp"], SAN_FLAGS, include=[SOLUTIONS, CHECKS])
    assert run(exe).split() == "PASS C12 PASS C15 PASS C18 PASS C20".split()


def test_concurrent_solutions_under_thread_sanitizer() -> None:
    exe = compile_cpp([CHECKS / "check_threads.cpp"], ["-fsanitize=thread"], include=[SOLUTIONS, CHECKS])
    assert run(exe).split() == "PASS C13 PASS C14 PASS C16 PASS C17 PASS C19".split()


def test_every_solution_header_is_exercised() -> None:
    used = set()
    for src in CHECKS.glob("*.cpp"):
        used |= set(re.findall(r'#include "(\w+\.hpp)"', src.read_text("utf-8")))
    headers = {p.name for p in SOLUTIONS.glob("*.hpp")}
    # thread_pool.hpp pulls in blocking_queue.hpp; both are included directly too
    assert headers <= used, headers - used
