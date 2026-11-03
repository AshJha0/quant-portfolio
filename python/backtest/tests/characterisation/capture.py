"""Capture the characterisation corpus of the pre-qp_bt backtesters (Feathers).

Run once, on the code *before* migration (commit r8x/r10-l2-backtest~ before
the migration commits), from the repository root:

    PYTHONPATH="python/core/src;python/foundations/02-trading-signal-backtest/src;\
python/equity/05-pairs-trading/src;python/equity/07-portfolio-optimization/src;\
python/equity/08-algo-execution/src;python/equity/10-regime-switching/src" \
    python python/backtest/tests/characterisation/capture.py

It writes ``corpus.npz`` next to this file. ``test_characterisation.py``
compares the migrated modules with it.
"""

from __future__ import annotations

import pathlib
import warnings

import cases
import numpy as np

OUT = pathlib.Path(__file__).with_name("corpus.npz")


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in cases.ALL.items():
            for key, val in fn().items():
                arrays[f"{name}::{key}"] = np.asarray(val, dtype=float)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {len(arrays)} arrays to {OUT}")


if __name__ == "__main__":
    main()
