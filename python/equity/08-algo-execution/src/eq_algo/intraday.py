"""Synthetic intraday market simulator with temporary + permanent impact.

Model (per trading day of ``n_buckets`` equal-length buckets):

- **Volume**: expected bucket volumes follow a U-shaped profile
  ``p_j  propto  1 + c * (2u_j - 1)^2`` (``u_j`` = bucket midpoint in [0,1]),
  normalised to sum to 1; realised volumes add seeded lognormal noise.
- **Mid price**: geometric random walk with per-bucket vol
  ``sigma_daily / sqrt(n_buckets)`` plus **permanent impact** of our own
  trades, *linear in participation* (Almgren-Chriss):
  ``dS_perm_j = side * perm_coef * sigma_daily * (q_j / V_day) * S_j``.
- **Execution price** of a market order in bucket ``j`` (LOB-lite fill
  model: market orders always fill, crossing the spread):
  ``fill_j = S_j * (1 + side * (half_spread + temp_j))`` with **temporary
  impact** following the empirical square-root law,
  ``temp_j = temp_coef * sigma_daily * sqrt(q_j / V_day)``.
  Temporary impact affects only the fill price and fully reverts — the next
  bucket's mid does not carry it (tested).

Why linear-permanent + square-root-temporary: linear permanent impact is the
only shape consistent with no dynamic arbitrage (Huberman-Stanzl), while
measured *temporary* cost across venues and horizons follows the square-root
law ``cost ~ sigma * sqrt(Q/V)`` (Almgren et al. 2005; Toth et al. 2011).
See docs/METHODOLOGY.md for alternatives (Obizhaeva-Wang resilience).

Everything stochastic takes an explicit seed / Generator; identical seeds
give bit-identical days (tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["u_shaped_profile", "IntradayConfig", "ExecutionResult", "IntradayMarket"]


def u_shaped_profile(n_buckets: int, curvature: float = 3.0) -> np.ndarray:
    """U-shaped intraday volume profile, normalised to sum to exactly 1.

    ``p_j propto 1 + curvature * (2 u_j - 1)^2`` with ``u_j = (j+0.5)/n``.
    ``curvature > 0`` guarantees open/close buckets are heavier than midday.
    """
    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    if curvature < 0:
        raise ValueError("curvature must be >= 0")
    u = (np.arange(n_buckets) + 0.5) / n_buckets
    w = 1.0 + curvature * (2.0 * u - 1.0) ** 2
    return w / w.sum()


@dataclass(frozen=True)
class IntradayConfig:
    """Parameters of the intraday simulator.

    Units: ``mid0`` in currency; ``day_volume`` in shares (expected);
    ``sigma_daily`` is the close-to-close daily vol as a *fraction* of price;
    ``spread_bps`` is the full quoted spread in basis points; the impact
    coefficients are dimensionless multipliers on ``sigma_daily``.
    """

    mid0: float = 100.0
    day_volume: float = 1_000_000.0
    n_buckets: int = 26
    sigma_daily: float = 0.02
    spread_bps: float = 5.0
    temp_coef: float = 0.3
    perm_coef: float = 0.5
    vol_noise: float = 0.0
    curvature: float = 3.0
    price_noise: float | None = None

    def __post_init__(self) -> None:
        if self.mid0 <= 0 or self.day_volume <= 0:
            raise ValueError("mid0 and day_volume must be > 0")
        if self.n_buckets < 1:
            raise ValueError("n_buckets must be >= 1")
        if self.sigma_daily < 0 or self.spread_bps < 0:
            raise ValueError("sigma_daily and spread_bps must be >= 0")
        if self.temp_coef < 0 or self.perm_coef < 0 or self.vol_noise < 0:
            raise ValueError("impact/noise coefficients must be >= 0")
        if self.price_noise is not None and self.price_noise < 0:
            raise ValueError("price_noise must be >= 0 when set")

    @property
    def bucket_noise(self) -> float:
        """Per-bucket mid-price noise vol; defaults to ``sigma_daily/sqrt(n)``.

        Setting ``price_noise=0`` gives a deterministic mid path while
        keeping impact scales tied to ``sigma_daily`` — used to isolate
        permanent vs temporary impact in tests.
        """
        if self.price_noise is not None:
            return self.price_noise
        return self.sigma_daily / np.sqrt(self.n_buckets)

    @property
    def profile(self) -> np.ndarray:
        return u_shaped_profile(self.n_buckets, self.curvature)


@dataclass
class ExecutionResult:
    """One simulated day of executing a parent order.

    ``fills`` columns: ``mid`` (pre-trade mid of the bucket), ``qty``,
    ``price`` (fill price), ``market_volume``, ``half_spread_cost`` and
    ``temp_cost`` (per-share, currency), ``perm_move`` (permanent mid move
    caused by the bucket's trade, currency).  ``mids`` has length
    ``n_buckets + 1`` (includes the post-close mid).
    """

    side: int
    parent_qty: float
    decision_price: float
    fills: pd.DataFrame = field(repr=False)
    mids: np.ndarray = field(repr=False)

    @property
    def filled_qty(self) -> float:
        return float(self.fills["qty"].sum())

    @property
    def avg_price(self) -> float:
        q = self.filled_qty
        if q == 0:
            raise ValueError("no shares filled; average price undefined")
        return float((self.fills["qty"] * self.fills["price"]).sum() / q)

    @property
    def arrival_price(self) -> float:
        return float(self.mids[0])

    @property
    def final_price(self) -> float:
        return float(self.mids[-1])


class IntradayMarket:
    """Seeded intraday market simulator (see module docstring for the model)."""

    def __init__(self, config: IntradayConfig | None = None) -> None:
        self.config = config or IntradayConfig()

    def sample_volumes(self, rng: np.random.Generator) -> np.ndarray:
        """Realised bucket volumes: profile * day volume * lognormal noise."""
        cfg = self.config
        base = cfg.profile * cfg.day_volume
        if cfg.vol_noise == 0.0:
            return base.copy()
        noise = rng.lognormal(mean=-0.5 * cfg.vol_noise**2, sigma=cfg.vol_noise,
                              size=cfg.n_buckets)
        return base * noise

    def execute(self, schedule: np.ndarray, side: int = 1,
                seed: int | np.random.Generator = 0,
                decision_price: float | None = None,
                market_volumes: np.ndarray | None = None) -> ExecutionResult:
        """Execute a child-order schedule with market orders, one per bucket.

        Parameters
        ----------
        schedule : array of length ``n_buckets``
            Non-negative share quantities per bucket (0 = no order).
        side : int
            +1 buy, -1 sell.
        seed : int or Generator
            Drives the market noise (price path + volumes).
        decision_price : float, optional
            Price at the investment decision (for TCA delay cost); defaults
            to the arrival mid ``mid0``.
        market_volumes : array, optional
            Override the sampled bucket volumes with a known tape (replay /
            hand-computed benchmark tests).

        Raises
        ------
        ValueError
            On negative quantities, wrong length, an order routed to a
            zero-volume bucket (nothing to trade against), or a child order
            exceeding the bucket's market volume (cannot be more than 100%
            of the printed volume — split across buckets or days).
        """
        cfg = self.config
        q = np.asarray(schedule, dtype=float)
        if q.shape != (cfg.n_buckets,):
            raise ValueError(f"schedule must have length n_buckets={cfg.n_buckets}")
        if not np.all(np.isfinite(q)):
            # NaN would pass every comparison below (NaN < 0 is False) and
            # silently produce NaN fill prices and a NaN average price.
            raise ValueError("schedule contains NaN or infinite quantities")
        if np.any(q < 0):
            raise ValueError("schedule quantities must be >= 0")
        if side not in (1, -1):
            raise ValueError("side must be +1 (buy) or -1 (sell)")
        rng = np.random.default_rng(seed) if not isinstance(seed, np.random.Generator) else seed

        if market_volumes is not None:
            volumes = np.asarray(market_volumes, dtype=float)
            if volumes.shape != (cfg.n_buckets,):
                raise ValueError(f"market_volumes must have length {cfg.n_buckets}")
            if not np.all(np.isfinite(volumes)):
                raise ValueError("market_volumes contain NaN or infinite values")
            if np.any(volumes < 0):
                raise ValueError("market_volumes must be >= 0")
            # still consume the volume-noise draws so the price path matches
            # the same-seed sampled day
            self.sample_volumes(rng)
        else:
            volumes = self.sample_volumes(rng)
        if np.any((volumes <= 0) & (q > 0)):
            j = int(np.argmax((volumes <= 0) & (q > 0)))
            raise ValueError(
                f"bucket {j} has zero market volume but the schedule routes "
                f"{q[j]:.0f} shares there; reschedule around the halt/auction"
            )
        if np.any(q > volumes):
            j = int(np.argmax(q > volumes))
            raise ValueError(
                f"bucket {j}: child order {q[j]:.0f} exceeds the bucket's market "
                f"volume {volumes[j]:.0f} (participation > 100%); split the "
                f"parent across more buckets or days"
            )
        sigma_b = cfg.bucket_noise
        half_spread_frac = cfg.spread_bps * 1e-4 / 2.0
        eps = rng.standard_normal(cfg.n_buckets)

        mids = np.empty(cfg.n_buckets + 1)
        mids[0] = cfg.mid0
        rows = []
        for j in range(cfg.n_buckets):
            mid = mids[j]
            if q[j] > 0:
                temp_frac = cfg.temp_coef * cfg.sigma_daily * np.sqrt(q[j] / cfg.day_volume)
                fill = mid * (1.0 + side * (half_spread_frac + temp_frac))
                perm_move = side * cfg.perm_coef * cfg.sigma_daily * (q[j] / cfg.day_volume) * mid
            else:
                temp_frac = 0.0
                fill = np.nan
                perm_move = 0.0
            rows.append({
                "mid": mid,
                "qty": q[j],
                "price": fill,
                "market_volume": volumes[j],
                "half_spread_cost": mid * half_spread_frac if q[j] > 0 else 0.0,
                "temp_cost": mid * temp_frac,
                "perm_move": perm_move,
            })
            mids[j + 1] = (mid + perm_move) * (1.0 + sigma_b * eps[j])
        fills = pd.DataFrame(rows)
        return ExecutionResult(
            side=side, parent_qty=float(q.sum()),
            decision_price=float(decision_price if decision_price is not None else cfg.mid0),
            fills=fills, mids=mids,
        )
