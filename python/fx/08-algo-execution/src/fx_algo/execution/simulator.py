"""24h OTC FX market simulator: session liquidity, impact, last-look.

Market-structure model (see docs/METHODOLOGY.md for the full register):

* **Session profiles** — spread, depth and vol are step functions of the
  hour-of-day (Asia / London / London-NY overlap / NY / late), taken from
  a :class:`~fx_algo.sessions.PairProfile`.  The London-NY overlap is the
  deepest and tightest window of the FX day.
* **Impact** — executing ``q`` (mm base) in a bucket with depth ``D``
  (mm base absorbed per bucket) and vol ``sigma`` (pips per bucket):

  - temporary (fill-only, fully decays next bucket):
    ``k_temp * sigma * sqrt(|q| / D)`` pips — the empirical square-root
    law, scaled by session depth;
  - permanent (shifts all later mids): ``k_perm * sigma * (|q| / D)``
    pips — linear, so the no-dynamic-arbitrage condition holds.

* **Venues** — a *firm-liquidity ECN* fills every order at the quoted
  price; a *last-look dealer* shows a tighter spread but holds the order
  and rejects with probability increasing in the price move against the
  dealer during the hold window (the post-GFC controversy codified in
  FX Global Code Principle 17).  Rejected orders are resubmitted at the
  post-move price plus the firm spread and a resubmit penalty.

All randomness flows through a single ``numpy.random.Generator`` drawn
in a fixed order, so two runs with the same seed share the same mid path
regardless of venue or schedule (common random numbers), and a zero
schedule reproduces :meth:`MarketSimulator.simulate_mids` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import expit

from ..sessions import PairProfile, make_time_grid

__all__ = [
    "last_look_reject_prob",
    "FirmVenue",
    "LastLookVenue",
    "ExecutionResult",
    "MarketSimulator",
]


def last_look_reject_prob(
    adverse_move_pips: np.ndarray | float,
    threshold_pips: float = 0.6,
    sharpness_pips: float = 0.2,
) -> np.ndarray:
    """Dealer rejection probability as a function of the hold-window move.

    ``adverse_move_pips`` is the mid move *in the client's trade
    direction* during the last-look hold window (positive = the dealer
    would fill at a now-stale, off-market price and lose).  The model is
    a logistic ramp

    .. math:: p = \\text{expit}((m - \\theta)/s),

    monotonically increasing in the adverse move: symmetric last-look
    would reject on |move|, but observed dealer behaviour (and the
    reason for the controversy) is asymmetric — moves in the dealer's
    favour are filled happily.

    Parameters
    ----------
    adverse_move_pips : array_like
        Move in pips, signed in the client's direction.
    threshold_pips : float
        Move at which rejection probability is 50%.
    sharpness_pips : float
        Logistic scale; smaller = harder cutoff.

    Returns
    -------
    numpy.ndarray
        Probabilities in (0, 1).
    """
    if sharpness_pips <= 0:
        raise ValueError(f"sharpness_pips must be > 0, got {sharpness_pips}")
    m = np.asarray(adverse_move_pips, dtype=float)
    return expit((m - threshold_pips) / sharpness_pips)


@dataclass(frozen=True)
class FirmVenue:
    """Firm-liquidity ECN: full quoted spread, zero rejections."""

    name: str = "firm-ecn"
    spread_mult: float = 1.0


@dataclass(frozen=True)
class LastLookVenue:
    """Last-look dealer stream: tighter quote, hold window, rejections.

    Attributes
    ----------
    spread_mult : float
        Quoted half-spread as a fraction of the firm venue's (< 1 — the
        lure of last-look liquidity).
    threshold_pips, sharpness_pips : float
        Parameters of :func:`last_look_reject_prob`.
    hold_seconds : float
        Length of the last-look hold window.  The mid diffusion seen in
        the window is the bucket innovation scaled by
        ``sqrt(hold_seconds / bucket_seconds)``; the client's *per-bucket
        alpha* (if any) is assumed to realise immediately after arrival,
        so informed flow is fully visible to the dealer inside the hold —
        this is the adverse-selection markout that makes flow "toxic".
    resubmit_penalty_pips : float
        Extra pips paid on the resubmitted order (crossing a fresh
        spread aggressively after a reject).
    """

    name: str = "last-look"
    spread_mult: float = 0.6
    threshold_pips: float = 0.6
    sharpness_pips: float = 0.2
    hold_seconds: float = 2.0
    resubmit_penalty_pips: float = 0.05


@dataclass
class ExecutionResult:
    """Full audit trail of one parent-order execution (TCA input).

    All price arrays are in price units; ``*_pips`` arrays in pips.
    ``mids_pre[t]`` is the mid an instant before the bucket-``t`` child
    order; ``perm_cum_pips[t]`` is the cumulative permanent impact
    already embedded in ``mids_pre[t]`` (signed, price direction).
    """

    pair: str
    pip_size: float
    side: int
    times_hours: np.ndarray
    dt_minutes: float
    qty: np.ndarray
    mids_pre: np.ndarray
    fills: np.ndarray
    quoted: np.ndarray
    rejected: np.ndarray
    temp_pips: np.ndarray
    perm_cum_pips: np.ndarray
    spread_pips: np.ndarray
    half_spread_pips: np.ndarray
    arrival_mid: float
    final_mid: float
    venue: str

    @property
    def total_qty(self) -> float:
        """Signed executed quantity (mm base)."""
        return float(self.qty.sum())

    @property
    def avg_fill(self) -> float:
        """Quantity-weighted average fill price."""
        x = np.abs(self.qty).sum()
        if x == 0:
            return float("nan")
        return float((np.abs(self.qty) * self.fills).sum() / x)

    @property
    def mids_path(self) -> np.ndarray:
        """Mid path of length n+1 (pre-trade mids plus the final mid)."""
        return np.concatenate([self.mids_pre, [self.final_mid]])

    @property
    def is_pips(self) -> float:
        """Implementation shortfall vs arrival mid, pips per unit traded."""
        x = np.abs(self.qty).sum()
        if x == 0:
            return 0.0
        return float(self.side * (self.avg_fill - self.arrival_mid) / self.pip_size)

    @property
    def rejection_rate(self) -> float:
        """Fraction of non-zero child orders rejected on first submit."""
        active = np.abs(self.qty) > 0
        if not active.any():
            return 0.0
        return float(self.rejected[active].mean())


class MarketSimulator:
    """Seeded 24h session-aware FX execution simulator.

    Parameters
    ----------
    profile : PairProfile
        Pair liquidity profile (spread/depth/vol by session).
    start_hour : float
        Absolute grid start hour (0 = London midnight).
    horizon_hours : float
        Grid horizon.
    dt_minutes : float
        Bucket length in minutes.
    k_temp : float
        Temporary (sqrt) impact coefficient, dimensionless.
    k_perm : float
        Permanent (linear) impact coefficient, dimensionless.
    max_participation : float
        Hard cap on ``|q_t| / depth_t`` per bucket; a child order above
        it raises ``ValueError`` (split across more buckets/sessions).
    vol_scale : float
        Multiplier on the session vol profile (0 gives a deterministic
        zero-vol path for cost decomposition tests).
    tradeable : numpy.ndarray of bool, optional
        Per-bucket tradeability mask (False = weekend/blackout: no mid
        moves and no fills allowed).
    """

    def __init__(
        self,
        profile: PairProfile,
        start_hour: float = 0.0,
        horizon_hours: float = 24.0,
        dt_minutes: float = 5.0,
        k_temp: float = 0.35,
        k_perm: float = 0.05,
        max_participation: float = 0.3,
        vol_scale: float = 1.0,
        tradeable: Optional[np.ndarray] = None,
    ) -> None:
        if k_temp < 0 or k_perm < 0:
            raise ValueError("impact coefficients must be >= 0")
        if not (0 < max_participation <= 1):
            raise ValueError(f"max_participation must be in (0,1], got {max_participation}")
        self.profile = profile
        self.dt_minutes = float(dt_minutes)
        self.times_hours = make_time_grid(start_hour, horizon_hours, dt_minutes)
        n = len(self.times_hours)
        mid_hours = self.times_hours + 0.5 * dt_minutes / 60.0
        self.spread_pips = profile.spread_pips_at(mid_hours)
        self.depth_bucket = profile.depth_at(mid_hours) * dt_minutes
        self.sigma_bucket_pips = vol_scale * profile.vol_at(mid_hours) * np.sqrt(dt_minutes)
        self.k_temp = k_temp
        self.k_perm = k_perm
        self.max_participation = max_participation
        if tradeable is None:
            tradeable = np.ones(n, dtype=bool)
        if len(tradeable) != n:
            raise ValueError("tradeable mask length must match the grid")
        self.tradeable = np.asarray(tradeable, dtype=bool)
        # market is closed on non-tradeable buckets: no diffusion
        self.sigma_bucket_pips = np.where(self.tradeable, self.sigma_bucket_pips, 0.0)

    @property
    def n_buckets(self) -> int:
        """Number of buckets in the grid."""
        return len(self.times_hours)

    def _draw(self, seed: int | np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        n = self.n_buckets
        z = rng.standard_normal(n)
        u = rng.uniform(size=n)
        return z, u

    def simulate_mids(self, seed: int | np.random.Generator) -> np.ndarray:
        """No-trade mid path (length n+1) for the given seed."""
        z, _ = self._draw(seed)
        inc = self.sigma_bucket_pips * z
        pip = self.profile.pip_size
        return self.profile.s0 + pip * np.concatenate([[0.0], np.cumsum(inc)])

    def execute(
        self,
        schedule: np.ndarray,
        venue: FirmVenue | LastLookVenue = FirmVenue(),
        seed: int | np.random.Generator = 0,
        alpha_pips_per_bucket: float = 0.0,
    ) -> ExecutionResult:
        """Execute a child-order schedule against the simulated market.

        Parameters
        ----------
        schedule : numpy.ndarray
            Signed child quantities per bucket (mm base), all the same
            sign; zeros allowed.  Sum is the parent quantity.
        venue : FirmVenue or LastLookVenue
            Liquidity source.
        seed : int or numpy.random.Generator
            Path seed (common random numbers across venues/schedules).
        alpha_pips_per_bucket : float
            Short-term alpha of the client's flow: deterministic mid
            drift per bucket in the trade direction.  This is what makes
            flow "toxic" to a last-look dealer (adverse selection).

        Returns
        -------
        ExecutionResult

        Raises
        ------
        ValueError
            On mixed-sign schedules, length mismatch, trading in a
            closed bucket, or a child order above the participation cap.
        """
        q = np.asarray(schedule, dtype=float)
        if len(q) != self.n_buckets:
            raise ValueError(
                f"schedule length {len(q)} != grid length {self.n_buckets}"
            )
        pos, neg = (q > 0).any(), (q < 0).any()
        if pos and neg:
            raise ValueError("schedule mixes buys and sells; one parent order per run")
        side = 1 if not neg else -1
        if np.any((np.abs(q) > 0) & ~self.tradeable):
            raise ValueError("schedule trades in a non-tradeable (weekend/blackout) bucket")
        cap = self.max_participation * self.depth_bucket
        over = np.abs(q) > cap + 1e-12
        if over.any():
            t = int(np.argmax(over))
            raise ValueError(
                f"child order {abs(q[t]):.1f}mm in bucket {t} exceeds the depth cap "
                f"{cap[t]:.1f}mm ({self.max_participation:.0%} of session depth) — "
                "split the parent across more buckets or sessions"
            )

        z, u = self._draw(seed)
        pip = self.profile.pip_size
        part = np.abs(q) / self.depth_bucket
        temp_pips = np.where(np.abs(q) > 0, self.k_temp * self.sigma_bucket_pips * np.sqrt(part), 0.0)
        perm_pips = side * self.k_perm * self.sigma_bucket_pips * part  # signed, price dir
        drift = alpha_pips_per_bucket * side * self.tradeable
        diffusion = self.sigma_bucket_pips * z
        inc = diffusion + drift  # exogenous move per bucket, pips

        move_before = np.concatenate([[0.0], np.cumsum(inc + perm_pips)[:-1]])
        perm_cum = np.concatenate([[0.0], np.cumsum(perm_pips)[:-1]])
        mids_pre = self.profile.s0 + pip * move_before
        final_mid = self.profile.s0 + pip * float(np.sum(inc + perm_pips))

        hs_firm = 0.5 * self.spread_pips
        if isinstance(venue, LastLookVenue):
            hs_quoted = venue.spread_mult * hs_firm
            quoted = mids_pre + side * pip * (hs_quoted + temp_pips)
            # Mid move inside the hold window: diffusion scaled down to
            # the hold horizon plus the flow's immediate (markout) alpha.
            hold_scale = np.sqrt(venue.hold_seconds / (60.0 * self.dt_minutes))
            hold_move = hold_scale * diffusion + drift  # pips, price direction
            adverse = side * hold_move  # pips, in the client's direction
            p = last_look_reject_prob(adverse, venue.threshold_pips, venue.sharpness_pips)
            rejected = (u < p) & (np.abs(q) > 0)
            refill = (
                mids_pre
                + pip * hold_move
                + side * pip * (hs_firm + temp_pips + venue.resubmit_penalty_pips)
            )
            fills = np.where(rejected, refill, quoted)
            hs_used = np.where(rejected, hs_firm + venue.resubmit_penalty_pips, hs_quoted)
        else:
            hs_quoted = venue.spread_mult * hs_firm
            quoted = mids_pre + side * pip * (hs_quoted + temp_pips)
            fills = quoted.copy()
            rejected = np.zeros(self.n_buckets, dtype=bool)
            hs_used = hs_quoted

        return ExecutionResult(
            pair=self.profile.name,
            pip_size=pip,
            side=side,
            times_hours=self.times_hours.copy(),
            dt_minutes=self.dt_minutes,
            qty=q.copy(),
            mids_pre=mids_pre,
            fills=fills,
            quoted=quoted,
            rejected=rejected,
            temp_pips=temp_pips,
            perm_cum_pips=perm_cum,
            spread_pips=self.spread_pips.copy(),
            half_spread_pips=np.broadcast_to(hs_used, (self.n_buckets,)).astype(float),
            arrival_mid=float(self.profile.s0),
            final_mid=final_mid,
            venue=getattr(venue, "name", "unknown"),
        )
