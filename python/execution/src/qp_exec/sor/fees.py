"""Venue fee schedules and latencies.

The data file is ``python/execution/venues/fee_schedules.toml`` (sources and
caveats are written at its top). ``DEFAULT_SCHEDULES`` mirrors it so the
package works when installed without the repository; a test keeps the two
identical.

Sign convention (dollars per share): ``maker > 0`` is a rebate to the
liquidity provider, ``taker > 0`` a fee to the liquidity taker. Inverted
venues have ``maker < 0`` and ``taker < 0``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

TICK_USD = 0.01  # one price tick, dollars per share


@dataclass(frozen=True)
class LitSchedule:
    name: str
    model: str
    maker: float
    taker: float
    latency_us: int

    @property
    def maker_ticks(self) -> float:
        """Maker rebate in ticks per share (cents for a one-cent tick)."""
        return self.maker / TICK_USD

    @property
    def taker_ticks(self) -> float:
        return self.taker / TICK_USD


@dataclass(frozen=True)
class DarkSchedule:
    name: str
    model: str
    fee: float
    latency_us: int
    min_qty: int
    interval_us: int = 0

    @property
    def fee_ticks(self) -> float:
        return self.fee / TICK_USD


DEFAULT_LIT: dict[str, LitSchedule] = {
    "LIT_A": LitSchedule("LIT_A", "maker-taker", 0.0020, 0.0030, 40),
    "LIT_B": LitSchedule("LIT_B", "maker-taker", 0.0015, 0.0028, 120),
    "LIT_C": LitSchedule("LIT_C", "inverted", -0.0010, -0.0005, 80),
    "LIT_D": LitSchedule("LIT_D", "flat", 0.0000, 0.0009, 250),
}
DEFAULT_DARK: dict[str, DarkSchedule] = {
    "DARK_MID": DarkSchedule("DARK_MID", "midpoint-continuous", 0.0010, 150, 0),
    "DARK_BLOCK": DarkSchedule("DARK_BLOCK", "midpoint-continuous", 0.0012, 300, 20),
    "DARK_COND": DarkSchedule("DARK_COND", "conditional", 0.0010, 300, 10),
    "DARK_FBA": DarkSchedule("DARK_FBA", "periodic-auction", 0.0008, 200, 0, 100_000),
}


def load_fee_table(path: Path) -> tuple[dict[str, LitSchedule], dict[str, DarkSchedule]]:
    """Read a fee table in the format of ``venues/fee_schedules.toml``."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    lit = {k: LitSchedule(k, v["model"], float(v["maker"]), float(v["taker"]),
                          int(v["latency_us"])) for k, v in raw.get("lit", {}).items()}
    dark = {k: DarkSchedule(k, v["model"], float(v["fee"]), int(v["latency_us"]),
                            int(v["min_qty"]), int(v.get("interval_us", 0)))
            for k, v in raw.get("dark", {}).items()}
    for s in lit.values():
        if s.taker > 0.003 + 1e-12:
            raise ValueError(f"{s.name}: taker fee above the Rule 610(c) $0.003 cap")
        if s.maker > s.taker and s.model == "maker-taker":
            raise ValueError(f"{s.name}: maker rebate above taker fee")
    return lit, dark
