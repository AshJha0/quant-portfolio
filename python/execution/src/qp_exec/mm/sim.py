"""Run a quoting model against the W25 reference book.

Plain idea
----------
A synthetic market: a fundamental price follows a tick random walk; a
background liquidity provider keeps a ladder of resting orders a few ticks
either side of it; takers send unit market orders at Poisson times. The
market maker reads the displayed top of book, asks its model for quotes,
passes them through ``QuoteController`` and posts post-only limit orders. It
is filled only by the book's own price-time matching, so queue position and
priority are real, not assumed.

Formal statement
----------------
The P&L is split per fill and per step, using the mark ``m`` (the last
displayed mid the maker saw)::

    spread capture  = sum over fills of sign * (m - price) * qty
    inventory P&L   = sum over steps of q * (m_new - m_old)
    total           = cash + q_T * m_T

and the two parts sum to the total exactly (the inventory sum telescopes).
The test checks this to 1e-9.

Only ``qp_exec.lob`` names from its published ``__init__`` are used: this is
the seam with the W25 lane.

Worked example
--------------
>>> from qp_exec.mm.quotes import ASParams
>>> r = run_mm(MMConfig(steps=200, seed=1), ASParams(gamma=0.1, sigma=0.05, k=20.0, q_max=5))
>>> abs(r.spread_pnl + r.inventory_pnl - r.total_pnl) < 1e-9
True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qp_exec.lob import Order, OrdType, ReferenceBook, Side, Tif

from .controls import ControlParams, QuoteController
from .quotes import GLFT, ASParams, GLFTParams, avellaneda_stoikov


@dataclass(frozen=True, slots=True)
class MMConfig:
    steps: int = 2000
    dt: float = 1.0
    tick: float = 0.01
    start_ticks: int = 10_000
    p_move: float = 0.3  # chance per step the fundamental moves one tick
    bg_offsets: tuple[int, ...] = (2, 3, 4, 5)  # background ladder, ticks from fundamental
    bg_qty: int = 5
    taker_rate: float = 0.8  # market orders per step per side
    horizon: float | None = None  # T for AS/finite GLFT; default steps * dt
    max_position: int = 5
    widen_at: int | None = 3
    max_updates: int | None = None
    window: float = 1.0
    stale_steps: frozenset[int] = field(default_factory=frozenset)  # feed flagged stale
    seed: int = 0


@dataclass
class MMResult:
    total_pnl: float = 0.0
    spread_pnl: float = 0.0
    inventory_pnl: float = 0.0
    fills: int = 0
    max_abs_q: int = 0
    final_q: int = 0
    quoted_while_stale: int = 0  # steps where an MM order rested during a stale flag
    stale_pulls: int = 0
    q_path: list[int] = field(default_factory=list)


def run_mm(cfg: MMConfig, model: ASParams | GLFTParams) -> MMResult:
    rng = np.random.default_rng(cfg.seed)
    book = ReferenceBook()
    ctl = QuoteController(ControlParams(cfg.tick, cfg.max_position, cfg.widen_at, 1,
                                        cfg.max_updates, cfg.window))
    glft = GLFT(model) if isinstance(model, GLFTParams) else None
    T = cfg.horizon if cfg.horizon is not None else cfg.steps * cfg.dt
    tick = cfg.tick

    next_id = 1
    bg_ids: list[int] = []
    mm_bid: int | None = None  # order ids
    mm_ask: int | None = None
    mm_px: dict[int, int] = {}
    fund = cfg.start_ticks
    q = 0
    cash = 0.0
    res = MMResult()
    mark: float | None = None
    ts = 0

    def nid() -> int:
        nonlocal next_id
        next_id += 1
        return next_id

    def book_fills(fills: tuple) -> None:  # type: ignore[type-arg]
        nonlocal q, cash, mm_bid, mm_ask
        for f in fills:
            if f.maker_id in mm_px:
                side = Side.BUY if f.maker_id == mm_bid else Side.SELL
                sgn = 1 if side == Side.BUY else -1
                px = f.price * tick
                m = mark if mark is not None else px
                cash -= sgn * px * f.qty
                res.spread_pnl += sgn * (m - px) * f.qty
                q += sgn * f.qty
                res.fills += 1
                if book.order_state(f.maker_id) is None:
                    mm_px.pop(f.maker_id, None)
                    if f.maker_id == mm_bid:
                        mm_bid = None
                    else:
                        mm_ask = None

    def cancel_mm(oid: int | None) -> None:
        if oid is not None:
            book.cancel(oid, ts)
            mm_px.pop(oid, None)

    for step in range(cfg.steps):
        ts += 1
        t = step * cfg.dt
        # 1. Fundamental moves; background ladder re-centres.
        if rng.random() < cfg.p_move:
            fund += 1 if rng.random() < 0.5 else -1
        for oid in bg_ids:
            book.cancel(oid, ts)
        bg_ids.clear()
        for off in cfg.bg_offsets:
            for side, px in ((Side.BUY, fund - off), (Side.SELL, fund + off)):
                o = Order(nid(), side, cfg.bg_qty, px, post_only=True)
                if book.add(o, ts).accepted:
                    bg_ids.append(o.order_id)
        # 2. Mark-to-market on the displayed mid.
        bb, ba = book.best_bid(), book.best_ask()
        # The mid includes the maker's own quotes, as the public feed shows them.
        if bb is not None and ba is not None and bb < ba:
            new_mark = 0.5 * (bb + ba) * tick
            if mark is not None:
                res.inventory_pnl += q * (new_mark - mark)
            mark = new_mark
        stale = step in cfg.stale_steps
        # 3. Model quote and controls.
        tau = max(T - t, 0.0)
        if mark is None:
            mb = ma = None
        elif glft is not None:
            qq = max(-glft.p.q_max, min(glft.p.q_max, q))
            mq = glft.quote(mark, qq, tau if cfg.horizon is not None else None)
            mb, ma = mq.bid, mq.ask
        else:
            assert isinstance(model, ASParams)
            mq = avellaneda_stoikov(mark, q, tau, model)
            mb, ma = mq.bid, mq.ask
        d = ctl.decide(mb, ma, q, bb, ba, stale, t)
        if d.reason == "stale":
            if mm_bid is not None or mm_ask is not None:
                res.stale_pulls += 1
        for want, cur_attr, side in ((d.bid, "bid", Side.BUY), (d.ask, "ask", Side.SELL)):
            cur = mm_bid if cur_attr == "bid" else mm_ask
            if cur is not None and (want is None or mm_px.get(cur) != want):
                cancel_mm(cur)
                cur = None
            if want is not None and cur is None:
                o = Order(nid(), side, 1, want, OrdType.LIMIT, Tif.DAY, post_only=True)
                r = book.add(o, ts)
                if r.accepted and r.resting_qty > 0:
                    cur = o.order_id
                    mm_px[cur] = want
            if cur_attr == "bid":
                mm_bid = cur
            else:
                mm_ask = cur
        if stale and (mm_bid is not None or mm_ask is not None):
            res.quoted_while_stale += 1
        # 4. Takers.
        for side in (Side.BUY, Side.SELL):
            for _ in range(rng.poisson(cfg.taker_rate)):
                r = book.add(Order(nid(), side, 1, 0, OrdType.MARKET, Tif.IOC), ts)
                book_fills(r.fills)
        res.max_abs_q = max(res.max_abs_q, abs(q))
        res.q_path.append(q)

    res.final_q = q
    res.total_pnl = cash + q * (mark if mark is not None else 0.0)
    return res
