"""Order flow, fills, markouts, impact fitting and stylised-fact bands."""

from __future__ import annotations

import numpy as np
import pytest
from qp_exec.lob import Order, Side
from qp_exec.lob.fills import markouts, simulate_passive_order
from qp_exec.lob.flow import ZIParams, queue_reactive, zero_intelligence
from qp_exec.lob.impact import (
    SIGN_ACF_EXPONENT_BAND,
    acf_exponent,
    fit_impact_exponent,
    lmf_signs,
    signature_ratio,
    window_imbalance,
)


@pytest.fixture(scope="module")
def zi():  # type: ignore[no-untyped-def]
    return zero_intelligence(ZIParams(), 20000, seed=1)


def test_zi_book_stays_valid_and_spread_distribution(zi) -> None:  # type: ignore[no-untyped-def]
    zi.book.check_invariants()
    sp = np.array(zi.spreads)
    assert sp.min() >= 1
    # spread distribution: mode at one or two ticks, mean within 1.5-3 ticks
    # for these rates (Cont, Stoikov and Talreja 2010 fig. 4 has the same shape)
    assert int(np.argmax(np.bincount(sp))) in (1, 2)
    assert 1.5 <= sp.mean() <= 3.0


def test_signature_plot_shows_bounce(zi) -> None:  # type: ignore[no-untyped-def]
    # Mid-quote mean reversion at short lags: ratio > 1. Band [1.2, 3].
    assert 1.2 <= signature_ratio(np.array(zi.mids)) <= 3.0


def test_queue_reactive_runs_and_tightens_spread() -> None:
    qr = queue_reactive(ZIParams(), 10000, seed=2)
    qr.book.check_invariants()
    assert np.mean(qr.spreads) < 2.0


def test_markouts_show_adverse_selection_on_passive_fills(zi) -> None:  # type: ignore[no-untyped-def]
    times = np.array(zi.times)
    mids = np.array(zi.mids)
    t_idx = {t: i for i, t in enumerate(zi.times)}
    ft, fp, sg, half = [], [], [], []
    for t, f in zi.trades:
        i = t_idx[t]
        if i == 0 or not np.isfinite(mids[i - 1]):
            continue
        s = 1 if f.aggressor == Side.SELL else -1  # maker's side
        ft.append(t)
        fp.append(f.price)
        sg.append(s)
        half.append(s * (mids[i - 1] - f.price))
    mo = markouts(np.array(ft), np.array(fp, float), np.array(sg), times, mids)
    se = np.nanstd(mo[:, 0]) / np.sqrt(len(mo))
    # Makers capture less than the half spread they posted: adverse selection.
    assert np.nanmean(mo[:, 0]) + 3 * se < np.mean(half)
    assert mo.shape[1] == 3


def test_latency_costs_queue_position(zi) -> None:  # type: ignore[no-untyped-def]
    ev, ts = zi.events[:4000], zi.times[:4000]
    t0 = ts[200]
    bb = None
    # join the best bid at decision time
    from qp_exec.lob import ReferenceBook
    from qp_exec.lob.events import apply_event
    b = ReferenceBook()
    for e, t in zip(ev, ts, strict=True):
        if t > t0:
            break
        apply_event(b, e)
    bb = b.best_bid()
    assert bb is not None
    fast = simulate_passive_order(ev, ts, Order(1, Side.BUY, 3, bb), t0, latency=0.0)
    slow = simulate_passive_order(ev, ts, Order(1, Side.BUY, 3, bb), t0, latency=30.0)
    assert fast.posted_at is not None and slow.posted_at is not None
    assert fast.queue_ahead[0][1] <= slow.queue_ahead[0][1] or len(fast.fills) >= len(slow.fills)
    filled = sum(f.qty for f in fast.fills)
    assert 0 <= filled <= 3


def test_sign_acf_exponent_in_named_band() -> None:
    s = lmf_signs(200_000, tail=1.5, n_active=20, seed=4)
    g, lo, hi = acf_exponent(s, lags=(10, 1000), n_boot=50)
    lo_b, hi_b = SIGN_ACF_EXPONENT_BAND
    assert lo_b <= g <= hi_b
    assert lo < g < hi
    # iid signs have no memory: the fitted ACF has no positive power law
    iid = np.random.default_rng(0).choice([-1, 1], 200_000)
    c = acf_exponent(iid, lags=(10, 1000), n_boot=0)[0]
    assert not (lo_b <= c <= hi_b) or np.isnan(c)


def test_impact_exponent_recovered_with_ci() -> None:
    rng = np.random.default_rng(3)
    q = rng.standard_t(3, 20000) * 50
    r = 0.02 * np.sign(q) * np.abs(q) ** 0.5 + rng.normal(0, 0.05, q.size)
    fit = fit_impact_exponent(q, r, n_boot=100)
    assert fit.ci[0] <= 0.5 <= fit.ci[1]
    assert abs(fit.delta - 0.5) < 0.08
    lin = fit_impact_exponent(q, 0.001 * q + rng.normal(0, 0.05, q.size), n_boot=100)
    assert lin.ci[0] > 0.5  # linear impact is told apart from square root


def test_window_imbalance_shapes() -> None:
    t = np.arange(10, dtype=float)
    sq = np.array([1, -1, 2, 2, 2, -1, 0, 1, 1, 1], float)
    px = np.arange(10, dtype=float)
    q, r = window_imbalance(t, sq, px, window=3.0)
    assert q.tolist() == [2.0, 3.0, 2.0] and r.tolist() == [3.0, 3.0, 3.0]
