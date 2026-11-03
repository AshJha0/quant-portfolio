"""Child process for the real kill test: trade slowly until killed."""

import sys
import time
from pathlib import Path

import conftest  # type: ignore[import-not-found]  # noqa: F401  (registers qp_oms)
from qp_oms.recovery import sim

seed, jpath, vpath = int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
acc = sim.Acceptor.load(seed, vpath)
orig = acc.new_order


def slow(*a):  # type: ignore[no-untyped-def]
    time.sleep(0.002)
    return orig(*a)


acc.new_order = slow  # type: ignore[method-assign]
print("ready", flush=True)
sim.run_trader(jpath, acc, seed, 400, None)
