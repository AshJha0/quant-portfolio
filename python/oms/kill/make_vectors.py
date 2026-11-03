"""Regenerate vectors.json. Review every diff by hand: it is the W30 parity oracle."""

import json
from pathlib import Path
from typing import Any

from qp_oms.kill import replay

P = ["QP", "RATES", "MM1", "UST10Y"]
CASES: list[dict[str, Any]] = [
    {
        "name": "worked_example",
        "commands": [
            {"op": "working", "id": 7, "path": P},
            {
                "op": "engage",
                "scope": ["QP", "RATES", "*", "*"],
                "actor": "alice",
                "reason": "desk loss limit",
            },
            {"op": "check", "path": ["QP", "RATES", "MM2", "UST2Y"]},
            {"op": "request_reset", "kill_id": 1, "actor": "alice"},
            {"op": "terminal", "id": 7},
            {"op": "request_reset", "kill_id": 1, "actor": "alice"},
            {"op": "approve_reset", "kill_id": 1, "actor": "alice"},
            {"op": "approve_reset", "kill_id": 1, "actor": "bob"},
            {"op": "check", "path": P},
        ],
    },
    {
        "name": "nesting_and_duplicate",
        "commands": [
            {"op": "engage", "scope": ["QP", "*", "*", "*"], "actor": "a"},
            {"op": "engage", "scope": P, "actor": "a"},
            {"op": "engage", "scope": ["QP", "*", "*", "*"], "actor": "a"},
            {"op": "check", "path": P},
            {"op": "request_reset", "kill_id": 1, "actor": "a"},
            {"op": "approve_reset", "kill_id": 1, "actor": "b"},
            {"op": "check", "path": P},
            {"op": "check", "path": ["QP", "RATES", "MM1", "UST2Y"]},
        ],
    },
    {
        "name": "cancel_on_disconnect",
        "commands": [
            {"op": "working", "id": 1, "path": P},
            {"op": "working", "id": 2, "path": ["QP", "RATES", "MM1", "UST2Y"]},
            {"op": "working", "id": 3, "path": ["QP", "FX", "MM9", "EURUSD"]},
            {"op": "disconnect", "paths": [P, ["QP", "RATES", "MM1", "UST2Y"]]},
            {"op": "terminal", "id": 2},
            {"op": "terminal", "id": 1},
            {"op": "check", "path": ["QP", "FX", "MM9", "EURUSD"]},
        ],
    },
]

if __name__ == "__main__":
    out = {
        "contract": "docs/KILL_CONTRACT.md",
        "version": 1,
        "cases": [dict(c, expected=replay(c["commands"])) for c in CASES],
    }
    (Path(__file__).parent / "vectors.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
