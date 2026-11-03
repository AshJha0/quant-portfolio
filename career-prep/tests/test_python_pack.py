"""Python pack: every snippet's output matches the key; solutions match brute force."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import random
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

PACK = Path(__file__).resolve().parents[1] / "python"
_spec = importlib.util.spec_from_file_location("cp_py_solutions", PACK / "solutions.py")
assert _spec and _spec.loader
sol = importlib.util.module_from_spec(_spec)
sys.modules["cp_py_solutions"] = sol
_spec.loader.exec_module(sol)

SNIPPET = re.compile(r"^### (Y\d\d) [^\n]*\n```python\n(.*?)```", re.M | re.S)
SNIPPETS = dict(SNIPPET.findall((PACK / "questions.md").read_text("utf-8")))
KEY = dict(re.findall(r"^- \*\*(Y\d\d)\*\* `([^`]+)`",
                      (PACK / "keys" / "answers.md").read_text("utf-8"), re.M))


def test_fourteen_snippets_each_keyed() -> None:
    assert len(SNIPPETS) == 14 and set(SNIPPETS) == set(KEY)


@pytest.mark.parametrize("pid", sorted(SNIPPETS))
def test_snippet_output_matches_key(pid: str) -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(SNIPPETS[pid], pid, "exec"), {"__name__": "__snippet__"})
    assert " | ".join(buf.getvalue().strip().splitlines()) == KEY[pid]


def test_max_drawdown() -> None:
    rng = random.Random(1)
    for _ in range(200):
        xs = [rng.uniform(1, 100) for _ in range(rng.randint(1, 30))]
        brute = max(((xs[i] - xs[j]) / xs[i] for i in range(len(xs))
                     for j in range(i, len(xs))), default=0.0)
        assert sol.max_drawdown(xs) == pytest.approx(max(brute, 0.0))


def test_vwap() -> None:
    assert sol.vwap([(10.0, 100), (11.0, 300)]) == pytest.approx(10.75)
    with pytest.raises(ValueError):
        sol.vwap([])


def test_rolling_max() -> None:
    rng = random.Random(3)
    for _ in range(200):
        xs = [rng.randint(-5, 5) for _ in range(rng.randint(1, 25))]
        k = rng.randint(1, len(xs))
        assert sol.rolling_max(xs, k) == [max(xs[i:i + k]) for i in range(len(xs) - k + 1)]


def test_lru_against_reference() -> None:
    rng = random.Random(4)
    cache = sol.LRUCache(3)
    ref: list[tuple[int, int]] = []  # most recent last
    for _ in range(2000):
        k = rng.randint(0, 6)
        if rng.random() < 0.5:
            hit = next((v for kk, v in ref if kk == k), None)
            if hit is not None:
                ref = [e for e in ref if e[0] != k] + [(k, hit)]
            assert cache.get(k) == hit
        else:
            v = rng.randint(0, 99)
            ref = [e for e in ref if e[0] != k] + [(k, v)]
            ref = ref[-3:]
            cache.put(k, v)


def test_merge_sorted() -> None:
    rng = random.Random(5)
    streams = [sorted((rng.randint(0, 50), float(s)) for _ in range(10)) for s in range(4)]
    out = list(sol.merge_sorted(streams))
    assert [t for t, _ in out] == sorted(t for st in streams for t, _ in st)
    assert sorted(out) == sorted(x for st in streams for x in st)


def test_asof_join_against_loop() -> None:
    rng = random.Random(6)
    qt = sorted(rng.sample(range(100), 20))
    tt = sorted(rng.sample(range(100), 15))
    quotes = pd.DataFrame({"time": qt, "bid": [float(i) for i in range(20)]})
    trades = pd.DataFrame({"time": tt})
    got = sol.asof_join(trades, quotes)["bid"].tolist()
    for t, b in zip(tt, got, strict=True):
        prior = [bid for q, bid in zip(qt, range(20), strict=True) if q <= t]
        if prior:
            assert b == prior[-1]
        else:
            assert pd.isna(b)
