"""Cross-language differential test: Python vs Rust vs C++ on seeded random inputs.

Each engine has a JSON-lines shim (``python -m eq_options._jsonl``, Rust
``eq_jsonl``, C++ ``eqopt_jsonl``). The same seeded requests go through
every shim that is available and each answer is compared with the Python
reference.

Locating the shims: ``QP_RS_EQ_JSONL`` and ``QP_CPP_EQ_JSONL`` hold a
command (it may be a prefix such as ``wsl /home/u/build/eq-gcc/eqopt_jsonl``).
Without ``QP_RS_EQ_JSONL`` the Rust release binary under
``rust/equity-options-engine/target/release`` is used if it exists. A
missing shim skips its test.

Tolerances. Every engine uses erfc-based normal CDFs and the same
formulas; outputs differ by libm rounding (a few ulps per call). A sum
of terms of magnitude M = max(S, K) with Gaussian-density factors whose
argument error is amplified by (1 + d1^2) gives
``64 eps (1 + d1^2) (|v| + M)``. Implied vol: ``4 max(eps M / vega,
1e-12 sigma)`` (one rounding of the quote over vega, plus the solver
floor). CRR: ``4 n eps M`` (n backward-induction steps of convex
combinations). Set ``QP_DIFF_VERBOSE=1`` to print max deviations.
"""

from __future__ import annotations

import json
import math
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from eq_options import _jsonl, bs_price

EPS = 2.220446049250313e-16
ROOT = Path(__file__).resolve().parents[4]


def _requests(seed: int = 20261101) -> list[dict]:
    rng = random.Random(seed)
    reqs: list[dict] = []
    for _ in range(300):
        S = rng.uniform(5, 500)
        reqs.append(dict(fn="bs", S=S, K=S * math.exp(rng.uniform(-1, 1)),
                         T=10 ** rng.uniform(-3, 1.5), r=rng.uniform(-0.02, 0.1),
                         q=rng.uniform(0, 0.06), sigma=10 ** rng.uniform(-2, 0.5),
                         type=rng.choice(["call", "put"])))
    for _ in range(200):
        S = rng.uniform(5, 500)
        K = S * math.exp(rng.uniform(-1.5, 1.5))
        T = 10 ** rng.uniform(-4, 1.5)
        r, q, sig = rng.uniform(-0.02, 0.1), rng.uniform(0, 0.06), 10 ** rng.uniform(-2.5, 0.7)
        t = rng.choice(["call", "put"])
        price = bs_price(S, K, T, r, sig, q, t)
        reqs.append(dict(fn="iv", price=price, S=S, K=K, T=T, r=r, q=q, type=t, _sigma=sig))
    for _ in range(40):
        S = rng.uniform(50, 150)
        reqs.append(dict(fn="crr", S=S, K=S * math.exp(rng.uniform(-0.4, 0.4)),
                         T=rng.uniform(0.1, 3), r=rng.uniform(-0.01, 0.08), q=rng.uniform(0, 0.05),
                         sigma=rng.uniform(0.1, 0.6), type=rng.choice(["call", "put"]),
                         american=rng.random() < 0.5, steps=rng.choice([50, 100, 200])))
    for _ in range(100):
        F = rng.uniform(5, 500)
        reqs.append(dict(fn="b76", F=F, K=F * math.exp(rng.uniform(-1, 1)),
                         T=10 ** rng.uniform(-3, 1), r=rng.uniform(-0.02, 0.1),
                         sigma=10 ** rng.uniform(-2, 0.3), type=rng.choice(["call", "put"])))
    for i, r in enumerate(reqs):
        r["id"] = i
    return reqs


def _wire(req: dict) -> str:
    return json.dumps({k: v for k, v in req.items() if not k.startswith("_")})


def _run_shim(cmd: list[str], reqs: list[dict]) -> list[dict]:
    text = "\n".join(_wire(r) for r in reqs) + "\n"
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
    assert len(lines) == len(reqs)
    return lines


def _shim(env: str, default: Path) -> list[str] | None:
    if os.environ.get(env):
        return shlex.split(os.environ[env])
    for cand in (default, default.with_suffix(".exe")):
        if cand.exists():
            return [str(cand)]
    return None


def _d1(req: dict) -> float:
    if req["fn"] == "b76":
        S, K, r, q = req["F"], req["K"], 0.0, 0.0
    else:
        S, K, r, q = req["S"], req["K"], req.get("r", 0.0), req.get("q", 0.0)
    sig = req.get("sigma", req.get("_sigma"))
    v = sig * math.sqrt(req["T"])
    return (math.log(S / K) + (r - q) * req["T"]) / v + 0.5 * v


def _tolerance(req: dict, name: str, ref: float, py_out: dict) -> float:
    M = max(req.get("S", req.get("F", 0.0)), req["K"])
    if req["fn"] == "iv":
        sig = py_out["sigma"]
        v = sig * math.sqrt(req["T"])
        d1 = (math.log(req["S"] / req["K"]) + (req["r"] - req["q"]) * req["T"]) / v + 0.5 * v
        vega = req["S"] * math.exp(-req["q"] * req["T"] - 0.5 * d1 * d1) * math.sqrt(req["T"] / (2 * math.pi))
        return 4.0 * max(EPS * M / vega if vega > 0 else math.inf, 1e-12 * sig)
    if req["fn"] == "crr":
        return 4.0 * req["steps"] * EPS * M
    d1 = _d1(req)
    return 64.0 * EPS * (1.0 + d1 * d1) * (abs(ref) + M)


def _compare(label: str, reqs: list[dict], got: list[dict]) -> None:
    worst: dict[tuple[str, str], tuple[float, float]] = {}
    failures = []
    for req, other in zip(reqs, got):
        py = json.loads(_jsonl.respond(_wire(req)))
        assert other["id"] == req["id"]
        if not py["ok"]:
            # Python refuses: the other engine must refuse too.
            assert not other["ok"], (label, req, other)
            continue
        if not other["ok"]:
            failures.append((req, "refused", other["err"]))
            continue
        for name, ref in py["out"].items():
            val = other["out"][name]
            if isinstance(ref, str) or isinstance(val, str):
                assert ref == val, (label, req, name, ref, val)
                continue
            dev = abs(val - ref)
            rel = dev / abs(ref) if ref else dev
            key = (req["fn"], name)
            a, b = worst.get(key, (0.0, 0.0))
            worst[key] = (max(a, dev), max(b, rel))
            if dev > _tolerance(req, name, ref, py["out"]):
                failures.append((req, name, ref, val, dev))
    if os.environ.get("QP_DIFF_VERBOSE"):
        for (fn, name), (dev, rel) in sorted(worst.items()):
            sys.stdout.write(f"{label} {fn:4s} {name:7s} max abs {dev:.3e} max rel {rel:.3e}\n")
    assert not failures, failures[:5]


@pytest.fixture(scope="module")
def requests_() -> list[dict]:
    return _requests()


def test_python_shim_protocol() -> None:
    good = json.loads(_jsonl.respond('{"id": 1, "fn": "bs", "S": 100, "K": 100, "T": 1, '
                                     '"r": 0.05, "sigma": 0.2, "q": 0, "type": "call"}'))
    assert good["ok"] and abs(good["out"]["price"] - 10.450583572185565) < 1e-12
    bad = json.loads(_jsonl.respond('{"id": 2, "fn": "bs", "S": -1, "K": 100, "T": 1, '
                                    '"r": 0.05, "sigma": 0.2, "q": 0, "type": "call"}'))
    assert bad == {"id": 2, "ok": False, "err": bad["err"]} and "S" in bad["err"]
    assert not json.loads(_jsonl.respond('{"id": 3, "fn": "nope"}'))["ok"]


def test_python_shim_as_subprocess(requests_) -> None:
    got = _run_shim([sys.executable, "-m", "eq_options._jsonl"], requests_[:20])
    _compare("python", requests_[:20], got)


def test_rust_matches_python(requests_) -> None:
    cmd = _shim("QP_RS_EQ_JSONL", ROOT / "rust/equity-options-engine/target/release/eq_jsonl")
    if cmd is None:
        pytest.skip("Rust shim not built (cargo build --release) and QP_RS_EQ_JSONL unset")
    _compare("rust", requests_, _run_shim(cmd, requests_))


def test_cpp_matches_python(requests_) -> None:
    cmd = _shim("QP_CPP_EQ_JSONL", ROOT / "cpp/equity-options-engine/build/eqopt_jsonl")
    if cmd is None:
        pytest.skip("C++ shim not built and QP_CPP_EQ_JSONL unset")
    _compare("cpp", requests_, _run_shim(cmd, requests_))


def test_iv_requests_recover_sigma(requests_) -> None:
    """The IV requests themselves are sound: Python recovers the input vol."""
    for req in (r for r in requests_ if r["fn"] == "iv"):
        py = json.loads(_jsonl.respond(_wire(req)))
        if py["ok"]:
            tol = _tolerance(req, "sigma", py["out"]["sigma"], py["out"])
            assert abs(py["out"]["sigma"] - req["_sigma"]) <= max(tol, 1e-9) or tol > 1e-6
