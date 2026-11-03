"""Generate pre-trade risk parity vectors from the Python reference (W28d).

The C++ gateway's RiskBook must give the same reject reason and the same
exposure numbers as python/oms/risk/pretrade.py on any operation sequence.
This script drives the reference with seeded random sequences and writes the
operations and expected outputs to tests/pretrade_vectors.json. Floats are
written with repr(), which round-trips exactly, so the C++ test compares
doubles with ==, not a tolerance.

Run from the repo root:  python cpp/ull/gateway/tools/make_pretrade_vectors.py
"""

from __future__ import annotations

import json
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
_pkg = types.ModuleType("qp_oms")
_pkg.__path__ = [str(ROOT / "python" / "oms")]
sys.modules.setdefault("qp_oms", _pkg)

from qp_oms.risk import Limits, OrderIntent, RiskBook, Side  # noqa: E402

LIMITS = dict(
    max_order_qty=100.0,
    max_order_notional=8_000.0,
    price_band=0.05,
    position_limit=300.0,
    credit_limit=15_000.0,
    throttle_msgs=8,
    throttle_window_ns=1_000,
    dup_window_ns=5_000,
)
REF = {"X": 100.0, "Y": 50.0, "Z": 0.1}


def run(seed: int, n: int) -> dict:
    rng = random.Random(seed)
    book = RiskBook(Limits(**LIMITS), ref_px=dict(REF), restricted={"R"})
    ops: list[dict] = []
    issued: list[str] = []
    now = 0
    last: dict | None = None
    for i in range(n):
        now += rng.choice([0, 1, 7, 50, 200, 900, 3000])
        kind = rng.choices(["new", "amend", "amend_result", "fill", "terminal"], [6, 2, 2, 2, 2])[0]
        if kind == "new" or not issued:
            sym = rng.choice(["X", "X", "Y", "Z", "R", "Q"])
            ref = REF.get(sym, 10.0)
            px = round(ref * (1 + rng.choice([-0.06, -0.02, -0.001, 0.0, 0.003, 0.02, 0.051])), 6)
            if rng.random() < 0.05:
                px = 0.0
            cid = rng.choice(issued) if issued and rng.random() < 0.05 else f"c{seed}-{i}"
            o = dict(
                cl_ord_id=cid,
                account=rng.choice(["A", "B"]),
                session=rng.choice(["S1", "S2"]),
                symbol=sym,
                side=rng.choice(["BUY", "SELL"]),
                qty=float(rng.choice([1, 5, 10, 25.5, 60, 99, 100, 101])),
                px=px,
                parent_id=rng.choice(["", "", "", "P1"]),
            )
            if last is not None and rng.random() < 0.1:
                o = {**last, "cl_ord_id": f"c{seed}-{i}"}  # content duplicate
            last = o
            r = book.check_new(OrderIntent(**{**o, "side": Side(o["side"])}), now)
            if r is None:
                issued.append(cid)
            op = {"op": "new", "now": now, "order": o, "expect": r}
        else:
            cid = rng.choice(issued)
            if kind == "amend":
                q = float(rng.choice([1, 10, 30, 80, 120]))
                op = {"op": "amend", "id": cid, "qty": q, "now": now, "expect": book.check_amend(cid, q, now)}
            elif kind == "amend_result":
                if cid not in book._working:
                    continue
                acc = rng.random() < 0.6
                book.on_amend_result(cid, acc)
                op = {"op": "amend_result", "id": cid, "accepted": acc}
            elif kind == "fill":
                if cid not in book._working:
                    continue
                q = float(rng.choice([1, 2.5, 10]))
                book.on_fill(cid, q)
                op = {"op": "fill", "id": cid, "qty": q}
            else:
                book.on_terminal(cid)
                op = {"op": "terminal", "id": cid}
        acct = rng.choice(["A", "B"])
        sym = rng.choice(["X", "Y", "Z"])
        lng, sht = book.worst_case(acct, sym)
        op["probe"] = {
            "account": acct,
            "symbol": sym,
            "long": repr(float(lng)),
            "short": repr(float(sht)),
            "notional": repr(float(book.working_notional(acct))),
        }
        ops.append(op)
    return {"seed": seed, "ops": ops}


def main() -> None:
    out = {
        "source": "python/oms/risk/pretrade.py via cpp/ull/gateway/tools/make_pretrade_vectors.py",
        "limits": LIMITS,
        "ref_px": REF,
        "restricted": ["R"],
        "cases": [run(s, 400) for s in (1, 2, 3, 4, 5)],
    }
    dest = Path(__file__).resolve().parents[1] / "tests" / "pretrade_vectors.json"
    dest.write_text(json.dumps(out, indent=0, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {dest} ({sum(len(c['ops']) for c in out['cases'])} ops)")


if __name__ == "__main__":
    main()
