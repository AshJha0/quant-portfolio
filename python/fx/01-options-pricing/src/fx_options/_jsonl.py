"""JSON-lines front end used by the cross-language differential test.

Run as ``python -m fx_options._jsonl``. Reads one flat JSON request per
stdin line and writes one response per line, in the same protocol as the
C++ ``fxopt_jsonl`` and Rust ``fx_jsonl`` shims. Functions: ``gk``,
``iv``, ``strike``, ``dns_atm``, ``crr``, ``fwd``.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

from .binomial import binomial_price
from .deltas import atm_dns_strike, delta, strike_from_delta
from .forwards import cip_forward
from .garman_kohlhagen import implied_vol
from .greeks import analytic_greeks


def _num(v: float) -> Any:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def handle(req: dict[str, Any]) -> dict[str, float]:
    """Evaluate one request; raise ValueError on bad input."""
    fn = req.get("fn")
    if fn == "gk":
        args = (req["S"], req["K"], req["T"], req["rd"], req["rf"], req["sigma"], req["type"])
        g = analytic_greeks(*args)
        return {
            "price": g.price, "delta_spot": g.delta_spot, "delta_fwd": g.delta_forward,
            "delta_spot_pa": delta(*args, "spot_pa"), "delta_fwd_pa": delta(*args, "forward_pa"),
            "gamma": g.gamma, "vega": g.vega, "theta": g.theta, "rho_d": g.rho_domestic,
            "rho_f": g.rho_foreign, "vanna": g.vanna, "volga": g.volga,
        }
    if fn == "iv":
        return {"sigma": implied_vol(req["price"], req["S"], req["K"], req["T"], req["rd"],
                                     req["rf"], req["type"])}
    if fn == "strike":
        return {"K": strike_from_delta(req["delta"], req["S"], req["T"], req["rd"], req["rf"],
                                       req["sigma"], req["type"], req["convention"])}
    if fn == "dns_atm":
        conv = "spot_pa" if req["pa"] else "spot"
        return {"K": atm_dns_strike(req["S"], req["T"], req["rd"], req["rf"], req["sigma"], conv)}
    if fn == "crr":
        exercise = "american" if req["american"] else "european"
        return {"price": binomial_price(req["S"], req["K"], req["T"], req["rd"], req["rf"],
                                        req["sigma"], req["type"], int(req["steps"]), exercise)}
    if fn == "fwd":
        return {"forward": cip_forward(req["S"], req["T"], req["rd"], req["rf"])}
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
