"""W27b studies: what the multi-venue simulation measures.

Every function runs seeded episodes and returns plain numbers with standard
errors, so tests can assert directions and the docs can quote values. The
model is stylised (see ``market.py``): the numbers illustrate mechanisms;
they are not estimates of any real venue.

* ``adverse_selection``: markout of our dark fills by counterparty type.
* ``min_qty_effect``: how a minimum quantity changes the fill distribution of
  a resting dark order (count, size, who we meet), with a predator pinging.
* ``ioi_leakage``: shortfall and predator detections with no IOIs,
  actionable IOIs and non-actionable IOIs.
* ``conditional_firm_up``: firm-up rates by participant on the conditional
  venue and whether the venue's penalty suspends the predator.
* ``score_learning``: one toxic dark venue; does markout learning move flow
  away from it, and what does that save?
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from qp_exec.lob import Side

from .dark import Predator
from .market import Market, MarketParams
from .router import SmartRouter
from .tca import VenueScores, markout

ALL_DARK = ("DARK_MID", "DARK_BLOCK", "DARK_COND", "DARK_FBA")
TAU = 1_000_000


@dataclass(frozen=True)
class Est:
    mean: float
    se: float
    n: int


def est(xs: list[float]) -> Est:
    n = len(xs)
    if n == 0:
        return Est(math.nan, math.nan, 0)
    mu = float(np.mean(xs))
    se = float(np.std(xs, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return Est(mu, se, n)


def adverse_selection(seeds: range, qty: int = 150, horizon_us: int = 4_000_000) -> dict[str, Est]:
    """1 s markout of dark fills by counterparty type: the lot-weighted mean
    per episode, then mean and standard error across episodes (fills within an
    episode share one price path, so lot-level errors would be too small)."""
    by: dict[str, list[float]] = defaultdict(list)  # one lot-weighted mean per episode
    for s in seeds:
        m = Market(s)
        side = Side.BUY if s % 2 == 0 else Side.SELL
        r = SmartRouter(m, side, qty, horizon_us, dark=ALL_DARK, seed=s).run()
        m.run_until(m.t + TAU + m.p.step_us)
        ep: dict[str, list[float]] = defaultdict(list)
        for f in r.fills:
            if f.liquidity == "D":
                ep[f.counterparty].extend([markout(f, m, TAU)] * f.qty)
        for k, v in ep.items():
            by[k].append(float(np.mean(v)))
    return {k: est(v) for k, v in sorted(by.items())}


@dataclass(frozen=True)
class MinQtyRow:
    min_qty: int
    p_any_fill: float
    fills_per_episode: Est
    fill_size: Est
    pinged_lots: Est  # lots of ours taken by 1-lot predator pings, per episode
    markout: Est  # per-episode lot-weighted 1 s markout, across episodes


def min_qty_effect(seeds: range, mqs: tuple[int, ...] = (0, 5, 20), qty: int = 60,
                   horizon_us: int = 3_000_000) -> list[MinQtyRow]:
    """Rest one dark buy of ``qty`` in DARK_MID for the horizon, per min-qty."""
    rows = []
    for mq in mqs:
        any_fill: list[float] = []
        nfills: list[float] = []
        sizes: list[float] = []
        pinged: list[float] = []
        mks: list[float] = []
        for s in seeds:
            m = Market(s, MarketParams(dark_rate=0.15))
            m.predators.append(Predator(m))
            m.run_until(50_000)
            m.send_dark("DARK_MID", Side.BUY, qty, mq)
            m.run_until(m.t + horizon_us)
            fs = [f for f in m.fills if f.venue == "DARK_MID"]
            m.run_until(m.t + TAU + m.p.step_us)
            any_fill.append(1.0 if fs else 0.0)
            nfills.append(float(len(fs)))
            sizes.extend(float(f.qty) for f in fs)
            pinged.append(float(sum(f.qty for f in fs if f.counterparty == "predator")))
            if fs:
                mks.append(float(np.mean([markout(f, m, TAU) for f in fs for _ in range(f.qty)])))
        rows.append(MinQtyRow(mq, float(np.mean(any_fill)), est(nfills), est(sizes),
                              est(pinged), est(mks)))
    return rows


@dataclass(frozen=True)
class IoiRow:
    mode: str
    shortfall: Est  # ticks per share
    detections: Est  # predator detections of our side per episode
    predator_pnl: Est  # ticks


def ioi_leakage(seeds: range, qty: int = 150, horizon_us: int = 4_000_000) -> list[IoiRow]:
    rows = []
    for mode in (None, "actionable", "non-actionable"):
        sf, det, pnl = [], [], []
        for s in seeds:
            m = Market(s, MarketParams(informed_share=0.1))
            pr = Predator(m, ping_prob=0.0)  # isolate the IOI channel
            m.predators.append(pr)
            r = SmartRouter(m, Side.BUY, qty, horizon_us, dark=("DARK_MID",), seed=s,
                            ioi=mode).run()
            m.run_until(m.t + pr.hold_steps * m.p.step_us + m.p.step_us)
            sf.append(r.shortfall)
            det.append(float(sum(1 for _, sd, _ in pr.detections if sd == Side.BUY)))
            pnl.append(pr.pnl_ticks)
        rows.append(IoiRow(mode or "none", est(sf), est(det), est(pnl)))
    return rows


@dataclass(frozen=True)
class FirmUp:
    rates: dict[str, Est]  # firm-up rate per participant, per episode
    predator_suspended: float  # share of episodes ending with the predator suspended
    router_fills_lots: Est


def conditional_firm_up(seeds: range, qty: int = 150, horizon_us: int = 4_000_000) -> FirmUp:
    rates: dict[str, list[float]] = defaultdict(list)
    susp, lots = [], []
    for s in seeds:
        m = Market(s, MarketParams(dark_rate=0.1))
        m.predators.append(Predator(m))
        r = SmartRouter(m, Side.BUY, qty, horizon_us, dark=("DARK_COND",), seed=s).run()
        dv = m.dark["DARK_COND"]
        for who, st in dv.firm_stats.items():
            if st.invites:
                rates[who].append(st.firms / st.invites)
        susp.append(1.0 if dv.suspended("predator") else 0.0)
        lots.append(float(sum(f.qty for f in r.fills if f.venue == "DARK_COND")))
    return FirmUp({k: est(v) for k, v in sorted(rates.items())}, float(np.mean(susp)), est(lots))


@dataclass(frozen=True)
class LearnRow:
    learning: bool
    toxic_share_late: Est  # share of dark lots sent to the toxic venue, second half
    shortfall: Est
    toxic_score: Est  # learned 1 s markout score of the toxic venue (learning only)


def score_learning(seeds: range, qty: int = 300, horizon_us: int = 8_000_000) -> list[LearnRow]:
    """DARK_MID is made toxic (90 % informed, 3x arrivals); DARK_BLOCK is clean."""
    rows = []
    for learn in (False, True):
        share, sf, score = [], [], []
        for s in seeds:
            m = Market(s)
            m.dark["DARK_MID"].informed_share = 0.9
            m.dark["DARK_MID"].rate_mult = 3.0
            m.dark["DARK_BLOCK"].informed_share = 0.0
            m.dark["DARK_BLOCK"].rate_mult = 3.0
            vs = VenueScores(tau_us=200_000, halflife_lots=50) if learn else None
            rt = SmartRouter(m, Side.BUY if s % 2 == 0 else Side.SELL, qty, horizon_us,
                             dark=("DARK_MID", "DARK_BLOCK"), seed=s, scores=vs)
            r = rt.run()
            half = len(rt.decisions) // 2
            tox = clean = 0.0
            for d in rt.decisions[half:]:
                for v, lim in zip(d["venues"], d["alloc"].limits, strict=True):
                    if v == "DARK_MID":
                        tox += lim
                    elif v == "DARK_BLOCK":
                        clean += lim
            if tox + clean > 0:
                share.append(tox / (tox + clean))
            sf.append(r.shortfall)
            if vs is not None:
                score.append(vs.score.get("DARK_MID", 0.0))
        rows.append(LearnRow(learn, est(share), est(sf), est(score)))
    return rows
