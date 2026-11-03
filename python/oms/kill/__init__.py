"""Kill-switch hierarchy (W28d). Contract: docs/KILL_CONTRACT.md."""

from .contract import (
    WILD,
    Kill,
    KillEvent,
    KillState,
    KillSwitch,
    Level,
    covers,
    make_scope,
    replay,
    validate_scope,
)

__all__ = [
    "WILD",
    "Kill",
    "KillEvent",
    "KillState",
    "KillSwitch",
    "Level",
    "covers",
    "make_scope",
    "replay",
    "validate_scope",
]
