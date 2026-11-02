"""Cross-language differential test: Python vs Rust vs C++ on seeded random inputs.

Each engine has a JSON-lines shim (``python -m fx_options._jsonl``, Rust
``fx_jsonl``, C++ ``fxopt_jsonl``). The same seeded requests go through
every shim that is available and each answer is compared with the Python
reference.

Locating the shims: ``QP_RS_FX_JSONL`` and ``QP_CPP_FX_JSONL`` hold a
command (it may be a prefix such as ``wsl /home/u/build/fx-gcc/fxopt_jsonl``).
Without ``QP_RS_FX_JSONL`` the Rust release binary under
``rust/target/release`` (the Cargo workspace target directory) is used if
it exists. The C++ shim defaults to ``cpp/<engine>/build/`` and then
``cpp/<engine>/build/gcc/`` (the gcc preset). A
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
from fx_options import _jsonl, gk_price

EPS = 2.220446049250313e-16
ROOT = Path(__file__).resolve().parents[4]


def _requests(seed: int = 20261101) -> list[dict]:
    rng = random.Random(seed)
    reqs: list[dict] = []
    def spot():  # EURUSD- or USDJPY-like
        return rng.choice([rng.uniform(0.5, 2.0), rng.uniform(80, 200)])

    for _ in range(300):
        S = spot()
        reqs.append(dict(fn="gk", S=S, K=S * math.exp(rng.uniform(-0.8, 0.8)),
                         T=10 ** rng.uniform(-3, 1.3), rd=rng.uniform(-0.01, 0.08),
                         rf=rng.uniform(-0.01, 0.08), sigma=10 ** rng.uniform(-2, 0.3),
                         type=rng.choice(["call", "put"])))
    for _ in range(200):
        S = spot()
        K = S * math.exp(rng.uniform(-1.0, 1.0))
        T = 10 ** rng.uniform(-4, 1.3)
        rd, rf, sig = rng.uniform(-0.01, 0.08), rng.uniform(-0.01, 0.08), 10 ** rng.uniform(-2.5, 0.5)
        t = rng.choice(["call", "put"])
        reqs.append(dict(fn="iv", price=gk_price(S, K, T, rd, rf, sig, t), S=S, K=K, T=T, rd=rd,
                         rf=rf, type=t, _sigma=sig))
    for _ in range(150):
        t = rng.choice(["call", "put"])
        d = rng.uniform(0.05, 0.45)
        reqs.append(dict(fn="strike", delta=d if t == "call" else -d, S=spot(),
                         T=10 ** rng.uniform(-2, 1), rd=rng.uniform(-0.01, 0.08),
                         rf=rng.uniform(-0.01, 0.08), sigma=rng.uniform(0.03, 0.5), type=t,
                         convention=rng.choice(["spot", "forward", "spot_pa", "forward_pa"])))
    for _ in range(50):
        reqs.append(dict(fn="dns_atm", S=spot(), T=10 ** rng.uniform(-2, 1),
                         rd=rng.uniform(-0.01, 0.08), rf=rng.uniform(-0.01, 0.08),
                         sigma=rng.uniform(0.03, 0.5), pa=rng.random() < 0.5))
    for _ in range(40):
        S = spot()
        reqs.append(dict(fn="crr", S=S, K=S * math.exp(rng.uniform(-0.3, 0.3)),
                         T=rng.uniform(0.1, 3), rd=rng.uniform(-0.01, 0.08), rf=rng.uniform(-0.01, 0.08),
                         sigma=rng.uniform(0.05, 0.4), type=rng.choice(["call", "put"]),
                         american=rng.random() < 0.5, steps=rng.choice([50, 100, 200])))
    for _ in range(50):
        reqs.append(dict(fn="fwd", S=spot(), T=10 ** rng.uniform(-2, 1.3),
                         rd=rng.uniform(-0.01, 0.08), rf=rng.uniform(-0.01, 0.08)))
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


def _shim(env: str, *defaults: Path) -> list[str] | None:
    if os.environ.get(env):
        return shlex.split(os.environ[env])
    for default in defaults:
        for cand in (default, default.with_suffix(".exe")):
            if cand.exists():
                return [str(cand)]
    return None


def _tolerance(req: dict, name: str, ref: float, py_out: dict) -> float:
    fn = req["fn"]
    M = max(req["S"], req.get("K", 0.0), abs(ref) if fn in ("strike", "dns_atm", "fwd") else 0.0)
    if fn == "fwd":
        # S e^{(rd - rf) T}: exp of an argument with relative error eps, times S.
        return 8.0 * EPS * M * (1.0 + abs((req["rd"] - req["rf"]) * req["T"]))
    if fn == "dns_atm":
        return 16.0 * EPS * M * (1.0 + req["sigma"] ** 2 * req["T"])
    if fn == "strike":
        # Unadjusted: K = F exp(-z v + v^2/2) with z from an inverse CDF good to
        # ~1e-15 absolute; dK/dz = K v. PA: Brent to xtol 1e-14 plus the
        # flatness of the PA delta in K near the fold. Bound: 1e-12 relative.
        return 1e-12 * M
    if fn == "iv":
        sig = py_out["sigma"]
        if sig == 0.0:
            return 0.0
        v = sig * math.sqrt(req["T"])
        d1 = (math.log(req["S"] / req["K"]) + (req["rd"] - req["rf"]) * req["T"]) / v + 0.5 * v
        vega = req["S"] * math.exp(-req["rf"] * req["T"] - 0.5 * d1 * d1) * math.sqrt(req["T"] / (2 * math.pi))
        return 4.0 * max(EPS * M / vega if vega > 0 else math.inf, 1e-12 * sig)
    if fn == "crr":
        return 4.0 * req["steps"] * EPS * M
    v = req["sigma"] * math.sqrt(req["T"])
    d1 = (math.log(req["S"] / req["K"]) + (req["rd"] - req["rf"]) * req["T"]) / v + 0.5 * v
    return 64.0 * EPS * (1.0 + d1 * d1) * (abs(ref) + M)


def _compare(label: str, reqs: list[dict], got: list[dict]) -> None:
    worst: dict[tuple[str, str], tuple[float, float]] = {}
    failures = []
    for req, other in zip(reqs, got, strict=False):
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
    good = json.loads(_jsonl.respond('{"id": 1, "fn": "gk", "S": 100, "K": 100, "T": 1, '
                                     '"rd": 0.05, "rf": 0, "sigma": 0.2, "type": "call"}'))
    assert good["ok"] and abs(good["out"]["price"] - 10.450583572185565) < 1e-12
    bad = json.loads(_jsonl.respond('{"id": 2, "fn": "gk", "S": -1, "K": 100, "T": 1, '
                                    '"rd": 0.05, "rf": 0, "sigma": 0.2, "type": "call"}'))
    assert bad == {"id": 2, "ok": False, "err": bad["err"]} and "S" in bad["err"]
    assert not json.loads(_jsonl.respond('{"id": 3, "fn": "nope"}'))["ok"]


def test_python_shim_as_subprocess(requests_) -> None:
    got = _run_shim([sys.executable, "-m", "fx_options._jsonl"], requests_[:20])
    _compare("python", requests_[:20], got)


def test_rust_matches_python(requests_) -> None:
    cmd = _shim("QP_RS_FX_JSONL", ROOT / "rust/target/release/fx_jsonl")
    if cmd is None:
        pytest.skip("Rust shim not built (cargo build --release) and QP_RS_FX_JSONL unset")
    _compare("rust", requests_, _run_shim(cmd, requests_))


def test_cpp_matches_python(requests_) -> None:
    cmd = _shim("QP_CPP_FX_JSONL",
        ROOT / "cpp/fx-options-engine/build/fxopt_jsonl",
        ROOT / "cpp/fx-options-engine/build/gcc/fxopt_jsonl",
    )
    if cmd is None:
        pytest.skip("C++ shim not built and QP_CPP_FX_JSONL unset")
    _compare("cpp", requests_, _run_shim(cmd, requests_))


def test_iv_requests_recover_sigma(requests_) -> None:
    """The IV requests themselves are sound: Python recovers the input vol."""
    for req in (r for r in requests_ if r["fn"] == "iv"):
        py = json.loads(_jsonl.respond(_wire(req)))
        if py["ok"] and py["out"]["sigma"] > 0.0:
            tol = _tolerance(req, "sigma", py["out"]["sigma"], py["out"])
            assert abs(py["out"]["sigma"] - req["_sigma"]) <= max(tol, 1e-9) or tol > 1e-6
