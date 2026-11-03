"""Migrated backtesters against the pre-migration corpus (W24 acceptance).

For each module the corpus either matches to 1e-12 or every difference is
listed in ``MIGRATION`` below and checked here: the difference is confined
to the columns and configurations it names, and stays inside a stated bound.
Hand fixtures for each listed change are in ``test_migration_fixtures.py``.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import cases  # noqa: E402

CORPUS = np.load(HERE / "corpus.npz")


def _new(module: str) -> dict[str, np.ndarray]:
    pytest.importorskip({"signal": "eq_signal_backtest", "port": "eq_port", "algo": "eq_algo",
                         "regime": "eq_regime", "pairs": "eq_pairs"}[module])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {k: np.asarray(v, dtype=float) for k, v in cases.ALL[module]().items()}


def _keys(module: str) -> list[str]:
    return [k.split("::", 1)[1] for k in CORPUS.files if k.startswith(module + "::")]


def _close(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> bool:
    if a.shape != b.shape:
        return False
    scale = max(1.0, float(np.nanmax(np.abs(b))) if b.size else 1.0)
    return bool(np.allclose(a, b, rtol=0, atol=tol * scale, equal_nan=True))


def test_pairs_matches_to_1e12():
    new = _new("pairs")
    bad = [k for k in _keys("pairs") if not _close(new[k], CORPUS[f"pairs::{k}"])]
    assert bad == []


def test_signal_positions_and_cost_free_returns_match_to_1e12():
    """eq_signal_backtest: positions, trade flags and every zero-cost return
    are unchanged. With costs only trade days move, by at most c (|r| + c):
    the cost is now paid on the notional at the fill (MIGRATION 1)."""
    new = _new("signal")
    for k in _keys("signal"):
        old = CORPUS[f"signal::{k}"]
        ek, c, what = k.split("|")
        cc = (float(c) + (10.0 if ek == "auction" else 0.0)) / 1e4   # + auction premium
        if what != "ret" or cc == 0.0:
            assert _close(new[k], old), k
            continue
        trades = CORPUS[f"signal::{ek}|{c}|trades"] > 0
        assert np.array_equal(new[k][~trades], old[~trades]), k
        p = cases._prices(1, 300, 1)["N00"].to_numpy()
        r = np.concatenate([[0.0], p[1:] / p[:-1] - 1])
        bound = cc * (np.abs(r) + cc) + 1e-15
        assert (np.abs(new[k] - old) <= bound)[trades].all(), k
        assert (new[k] != old)[trades].any(), k   # the change is real, not vacuous
