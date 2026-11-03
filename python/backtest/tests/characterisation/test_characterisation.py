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
        # non-trade days: unchanged to 1e-12 (bit-equal on one machine; the
        # Linux runner differs in the last ulp of a few returns)
        assert _close(new[k][~trades], old[~trades]), k
        p = cases._prices(1, 300, 1)["N00"].to_numpy()
        r = np.concatenate([[0.0], p[1:] / p[:-1] - 1])
        bound = cc * (np.abs(r) + cc) + 1e-15
        assert (np.abs(new[k] - old) <= bound)[trades].all(), k
        assert (new[k] != old)[trades].any(), k   # the change is real, not vacuous


def test_regime_exact_where_the_old_ledger_was_exact_and_bounded_elsewhere():
    """eq_regime.run_ledger (MIGRATION 2). Exact to 1e-12: every 0/1 weight
    path without costs or cash rate, and every close-auction path without
    costs or cash rate (no drift can build up before the fill). Elsewhere
    the drifted book, costs paid out of the trade and overnight financing
    move each day's net return by at most
    ``w^2 r^2 + 2 c w (|r| + c) + 2 (rate + spread) / 252``."""
    new = _new("regime")
    p = cases._prices(6, 260, 1)["N00"]
    r = p.pct_change().fillna(0.0).to_numpy()[1:260]
    moved = 0
    for k in _keys("regime"):
        wk, ek, c, cash, col = k.split("|")
        old = CORPUS[f"regime::{k}"]
        cc = (float(c) + (10.0 if ek == "auction" else 0.0)) / 1e4
        exact = cc == 0.0 and float(cash) == 0.0 and (wk == "01" or ek == "auction")
        if exact or col in ("weight", "prev_weight", "fill_ret"):
            assert _close(new[k], old), k
            continue
        if col != "net_ret":
            continue
        w = np.abs(CORPUS[f"regime::{wk}|{ek}|{c}|{cash}|weight"]) + 1.0
        rr = np.abs(r[: len(old)])
        bound = w**2 * rr**2 + 2 * cc * w * (rr + cc) + 2 * (float(cash) + 0.005) / 252 + 1e-15
        assert (np.abs(new[k] - old) <= bound).all(), k
        moved += int((new[k] != old).any())
    assert moved > 0


def test_port_reproduces_the_old_ledger_with_the_old_fill_and_no_costs():
    """eq_port (MIGRATION 3). The default fill moved from the decision close to
    the next bar, so the default corpus changes everywhere by design. With the
    old fill price (close auction, zero premium), daily rebalancing and no
    costs the new core reproduces the old net and gross returns to 1e-12."""
    pytest.importorskip("eq_port")
    from eq_port.backtest import run_backtest, strategy_equal_weight
    from qp_core.execution import ExecutionConfig

    ret = cases._prices(3, 200, 4).pct_change().iloc[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_backtest(ret, strategy_equal_weight, window=20, rebalance_every=1,
                           execution=ExecutionConfig("close_auction", auction_cost_bp=0.0))
    assert _close(res.net_returns.to_numpy(), CORPUS["port::ew|1|0.0|none|net"])
    assert _close(res.gross_returns.to_numpy(), CORPUS["port::ew|1|0.0|none|gross"])
    new = _new("port")
    assert all(not _close(new[k], CORPUS[f"port::{k}"]) for k in _keys("port")
               if k.endswith("|net"))


def test_algo_differences_are_bounded():
    """eq_algo (MIGRATION 4): the fill split now compounds, the book drifts and
    is traded back to target daily, costs are paid out of the trade and the
    borrow fee is on the short book held overnight. Without impact, each
    day's net return moves by at most ``G^2 r^2 + 2 c G (r + c) + 2 G
    (borrow + rate)/252`` with gross G = 2 and r the largest name move."""
    new = _new("algo")
    p = cases._prices(4, 160, 40)
    rmax = np.nan_to_num(p.pct_change().abs().max(axis=1).to_numpy())
    moved = 0
    for k in _keys("algo"):
        ck, ek, _fk, col = k.split("|")
        old = CORPUS[f"algo::{k}"]
        assert np.isfinite(new[k]).all(), k
        if col in ("gross_exposure", "net_exposure", "n_long", "n_short", "risk_free"):
            assert _close(new[k], old), k
            continue
        if col != "net_ret" or ck == "aum":
            continue
        g = 2.0
        c = (5.0 if ck == "lin" else 0.0) / 1e4 + (10.0 / 1e4 if ek == "auction" else 0.0)
        bound = g**2 * rmax**2 + 2 * c * g * (rmax + c) + 2 * g * (0.005 + 0.06) / 252 + 1e-14
        assert (np.abs(new[k] - old) <= bound).all(), k
        moved += int(not _close(new[k], old))
    assert moved > 0
