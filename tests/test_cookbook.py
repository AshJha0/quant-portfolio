"""Run every Python recipe in docs/COOKBOOK.md and check its quoted output.

Each ```python block is run in a fresh interpreter from the repository root,
with every workspace ``src`` folder and ``research`` on ``PYTHONPATH`` (no
install needed). The test then collects the numbers the recipe quotes as
output: decimals and comma-grouped integers inside backticks in the prose
after the block, in a following ```text block, and in ``#`` comments inside
the block. Every one of them must appear in what the recipe printed.

Some quoted numbers are inputs or rounded restatements, not printed output;
they are listed per recipe in ``NOT_PRINTED`` with the reason, so the list
stays short and reviewable.

A recipe whose third-party dependency is missing (for example the compiled
bindings, QuantLib or cvxpy) is skipped, not failed, and says which module.

Run from the repository root::

    python -m pytest -p no:cacheprovider -q tests/test_cookbook.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
COOKBOOK = REPO / "docs" / "COOKBOOK.md"

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.\d+(?:e[-+]?\d+)?|(?<![\w.])\d{1,3}(?:,\d{3})+")

# Heading prefix -> numbers quoted in the prose that the recipe does not print.
NOT_PRINTED: dict[str, set[str]] = {
    "How do I price an FX option via Garman-Kohlhagen": {"1.10", "1.0850"},  # inputs
    "How do I price an American option via the binomial": {"0.02"},  # input
    "How do I forecast next-period volatility with a GARCH": {"0.90", "0.1459", "0.05"},  # inputs, text
    "What happens at the edges (T": {"7.8077", "1.198e-54"},  # rounded restatements
    "What happens at the edges for Heston": {"0.01", "0.99"},  # inputs
    "How do I backtest a VaR series with Kupiec": {"0.01"},  # input
    "How do I bootstrap and query a yield curve": {"4.28", "4.02"},  # rounded
    "How do I price an FX forward from the spot date": {"1.106918", "10,000,000", "0.02", "5.80", "1.0850"},
    "How do I see fixed-income risk models break down": {"53,291", "25,243", "3,894", "162,817"},  # rounded
    "How do I test an FX pair for cointegration": {"1.88"},  # rounded
    "How do I estimate a PD model": {"0.766", "2.37", "0.437"},  # rounded
    "How do I compute expected loss and Basel IRB": {"291.2", "3.64", "2.89", "41.8"},  # in millions/billions
    "How do I run a minimum-variance / risk-parity": {"13.2", "21.9"},  # percent of weight
    "How do I run portfolio optimization for a basket": {"20.3", "19.2"},  # percent of weight
    "How do I see portfolio optimization break down": {"2.9e8", "1.0e-12"},  # rounded
    "How do I run the foundations no-look-ahead": {"0.0"},  # text
    "How do I walk-forward-validate": {"-1.59"},  # rounded
    "How do I price an American call on a stock with one": {"2.0", "0.5"},  # inputs
    "How do I quote both sides as a market maker": {"0.1", "1.5"},  # inputs
    "How do I compute the CVA of a swap": {"0.05", "14,295", "14,303"},  # SE ratio, rounded
    "How do I price a standard CDS on ISDA": {"379.1", "0.23"},  # integration check, not this run
    "How do I split an order across lit venues": {"1,059", "1,019", "1,011.9", "2,400", "0.8"},
    "How do I see what filling at the decision close": {"1.0175", "1.0074"},  # rounded
    "How do I shrink a covariance matrix": {"0.89", "1e-10"},
    "How do I attribute a return to the dollar": {"0.2", "0.8", "1e-10"},  # true betas, tolerance
    "How do I value a SOFR swap from dated legs": {"4.96", "7,200"},  # rounded
}
NEEDS_MODULE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")


def _recipes() -> list[tuple[str, int, str, str]]:
    text = COOKBOOK.read_text(encoding="utf-8").replace("\r\n", "\n")
    out = []
    for sec in re.split(r"^### ", text, flags=re.M)[1:]:
        head = sec.splitlines()[0]
        blocks = list(re.finditer(r"^```python\n(.*?)^```", sec, re.M | re.S))
        for k, b in enumerate(blocks):
            end = blocks[k + 1].start() if k + 1 < len(blocks) else len(sec)
            out.append((head, k, b.group(1), sec[b.end():end]))
    return out


def _quoted(code: str, after: str) -> set[str]:
    spans = re.findall(r"`([^`\n]+)`", after)
    spans += re.findall(r"^```text\n(.*?)^```", after, re.M | re.S)
    spans += [line.split("#", 1)[1] for line in code.splitlines() if "#" in line]
    return {n for s in spans for n in NUM.findall(s)}


def _env() -> dict[str, str]:
    srcs = sorted(REPO.glob("python/*/src")) + sorted(REPO.glob("python/*/*/src"))
    paths = [str(p) for p in srcs] + [str(REPO / "research")]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    return env


RECIPES = _recipes()


def test_recipe_count_is_plausible():
    assert len(RECIPES) >= 80, len(RECIPES)


@pytest.mark.parametrize(("head", "k", "code", "after"), RECIPES,
                         ids=[f"{h[:60]}#{k}" for h, k, _, _ in RECIPES])
def test_recipe_runs_and_prints_what_it_quotes(head, k, code, after):
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", env=_env(), cwd=REPO, timeout=900)
    if r.returncode != 0:
        m = NEEDS_MODULE.search(r.stderr)
        if m and not any((REPO / "python").glob(f"**/src/{m.group(1).split('.')[0]}")):
            pytest.skip(f"needs {m.group(1)}")
        pytest.fail(f"recipe raised:\n{r.stderr[-2000:]}")
    allowed = next((v for p, v in NOT_PRINTED.items() if head.startswith(p)), set())
    printed = r.stdout
    missing = sorted(n for n in _quoted(code, after) - allowed if n.lstrip("+") not in printed)
    assert not missing, f"quoted but not printed: {missing}\nstdout:\n{printed[-2000:]}"
