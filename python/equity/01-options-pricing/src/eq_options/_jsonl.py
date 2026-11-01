"""JSON-lines front end used by the cross-language differential test.

Run as ``python -m eq_options._jsonl``. Reads one flat JSON request per
stdin line and writes one response per line, in the same protocol as the
C++ ``eqopt_jsonl`` and Rust ``eq_jsonl`` shims:
``{"id": .., "ok": true, "out": {..}}`` or ``{"id": .., "ok": false, "err": ".."}``.
Functions: ``bs``, ``iv``, ``crr``, ``b76``.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from .binomial import crr_price
from .black76 import black76_greeks
from .black_scholes import implied_vol
from .greeks import bs_greeks


def _num(v: float) -> Any:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def handle(req: dict[str, Any]) -> dict[str, float]:
    """Evaluate one request; raise ValueError on bad input."""
    fn = req.get("fn")
    if fn == "bs":
        g = bs_greeks(req["S"], req["K"], req["T"], req["r"], req["sigma"], req["q"], req["type"])
        return {k: getattr(g, k) for k in ("price", "delta", "gamma", "vega", "theta", "rho",
                                           "vanna", "volga")}
    if fn == "iv":
        return {"sigma": implied_vol(req["price"], req["S"], req["K"], req["T"], req["r"],
                                     req["q"], req["type"])}
    if fn == "crr":
        exercise = "american" if req["american"] else "european"
        return {"price": crr_price(req["S"], req["K"], req["T"], req["r"], req["sigma"],
                                   req["q"], req["type"], exercise, int(req["steps"]))}
    if fn == "b76":
        g = black76_greeks(req["F"], req["K"], req["T"], req["r"], req["sigma"], req["type"])
        return {k: getattr(g, k) for k in ("price", "delta", "gamma", "vega", "theta", "rho")}
    raise ValueError(f"unknown fn: {fn!r}")


def respond(line: str) -> str:
    """Return the JSON response line for one request line."""
    rid: Any = None
    try:
        req = json.loads(line)
        rid = req.get("id")
        out = handle(req)
        return json.dumps({"id": rid, "ok": True, "out": {k: _num(v) for k, v in out.items()}})
    except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
        return json.dumps({"id": rid, "ok": False, "err": str(exc)})


def main() -> None:
    for line in sys.stdin:
        if line.strip():
            sys.stdout.write(respond(line) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
