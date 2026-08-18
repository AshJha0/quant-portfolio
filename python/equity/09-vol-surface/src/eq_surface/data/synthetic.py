"""Seeded synthetic equity option chains.

Two generation modes:

* ``mode="heston"`` -- prices generated from a *known* Heston parameter set
  (:data:`DEFAULT_TRUE_HESTON` by default) via the Fourier pricer.  This gives
  calibration a ground truth to recover, which is the only honest way to
  validate a calibrator.
* ``mode="svi"`` -- implied vols generated from hand-built arbitrage-free SVI
  slices with a realistic equity shape: mild put skew, smile curvature, skew
  flattening with maturity (term structure ~ 1/sqrt(T) skew decay).

Realism choices:

* Strike coverage narrows for short expiries: strikes are kept within
  ``F * exp(+-3.3 sigma_atm sqrt(T))`` intersected with 0.5-1.5 moneyness --
  listed markets do not quote 50%-moneyness weeklies.
* Optional bid/ask: quotes get a proportional half-spread that widens in the
  wings, and the mid is jittered inside the spread (seeded), mimicking
  microstructure noise in marks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..black_scholes import bs_price
from ..heston import HestonParams, heston_call_gl
from ..smile import SVIParams, svi_total_variance

__all__ = ["ChainData", "generate_chain", "DEFAULT_TRUE_HESTON", "default_svi_slices"]

#: Ground-truth parameter set used by ``mode="heston"`` (equity-index-like:
#: negative correlation, vol-of-vol high enough to violate Feller mildly).
DEFAULT_TRUE_HESTON = HestonParams(v0=0.035, kappa=1.8, theta=0.045, rho=-0.65, xi=0.45)

DEFAULT_EXPIRIES = (1.0 / 52.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0)


@dataclass
class ChainData:
    """A synthetic option chain plus its generation ground truth.

    Attributes
    ----------
    df : pandas.DataFrame
        Columns: ``expiry`` (years), ``strike``, ``forward``, ``moneyness``
        (K/F), ``log_moneyness`` (ln(K/F)), ``call_mid`` (present value),
        and when bid/ask noise is on, ``call_bid`` / ``call_ask``.
    spot, rate, div_yield : float
        Market inputs used to build the chain.
    mode : str
        ``"heston"`` or ``"svi"``.
    true_heston : HestonParams or None
        Ground-truth Heston parameters (``mode="heston"`` only).
    true_svi : dict or None
        Expiry -> SVIParams ground truth (``mode="svi"`` only).
    """

    df: pd.DataFrame
    spot: float
    rate: float
    div_yield: float
    mode: str
    true_heston: HestonParams | None = None
    true_svi: dict | None = None

    def slice(self, T: float) -> pd.DataFrame:
        """Return the chain rows for one expiry."""
        out = self.df[np.isclose(self.df["expiry"], T)]
        if out.empty:
            raise KeyError(f"no expiry {T} in chain; available: {sorted(self.df['expiry'].unique())}")
        return out


def default_svi_slices(expiries: np.ndarray) -> dict[float, SVIParams]:
    """Hand-built arbitrage-free SVI slices with equity-like term structure.

    ATM vol rises from ~19% (1w) towards ~21.5% (2y); skew (via rho and b)
    decays with maturity roughly like 1/sqrt(T).
    """
    slices = {}
    for T in np.asarray(expiries, dtype=float):
        atm_vol = 0.19 + 0.025 * (1.0 - np.exp(-0.8 * T))
        w_atm = atm_vol**2 * T
        b = 0.8 * w_atm / np.sqrt(T) / 0.19 * 0.35
        rho = -0.65 / (1.0 + 0.6 * np.sqrt(T))
        sig = 0.08 + 0.12 * np.sqrt(T)
        m = 0.05 * np.sqrt(T) * rho
        # solve a so that w(0) = w_atm
        d0 = -m
        a = w_atm - b * (rho * d0 + np.sqrt(d0 * d0 + sig * sig))
        p = SVIParams(a=float(a), b=float(b), rho=float(rho), m=float(m), sigma=float(sig))
        slices[float(T)] = p
    return slices


def generate_chain(
    mode: str = "heston",
    seed: int = 0,
    spot: float = 100.0,
    rate: float = 0.02,
    div_yield: float = 0.01,
    expiries: tuple = DEFAULT_EXPIRIES,
    moneyness_lo: float = 0.5,
    moneyness_hi: float = 1.5,
    moneyness_step: float = 0.05,
    bid_ask: bool = False,
    half_spread_atm: float = 0.002,
    true_heston: HestonParams | None = None,
) -> ChainData:
    """Generate a seeded synthetic equity option chain.

    Parameters
    ----------
    mode : {"heston", "svi"}
        Price source: known Heston parameters (calibration ground truth) or
        hand-built SVI slices.
    seed : int
        Seed for bid/ask jitter (chain is fully deterministic given the seed).
    spot, rate, div_yield : float
        Market inputs (continuous, annualised, ACT/365F).
    expiries : tuple
        Expiries in years (default 1w, 1m, 3m, 6m, 1y, 2y).
    moneyness_lo, moneyness_hi, moneyness_step : float
        Strike grid in forward moneyness K/F; per expiry the grid is clipped
        to ``exp(+-3.3 sigma_atm sqrt(T))`` for realism.
    bid_ask : bool
        Add proportional bid/ask around the mid (wing-widening) and jitter
        the mid inside the spread.
    half_spread_atm : float
        ATM half-spread as a fraction of the option price.
    true_heston : HestonParams, optional
        Override the ground-truth Heston set (``mode="heston"`` only).

    Returns
    -------
    ChainData
    """
    if mode not in ("heston", "svi"):
        raise ValueError(f"mode must be 'heston' or 'svi', got {mode!r}")
    if moneyness_lo <= 0.0 or moneyness_hi <= moneyness_lo:
        raise ValueError("need 0 < moneyness_lo < moneyness_hi")
    expiries_arr = np.sort(np.asarray(expiries, dtype=float))
    if np.any(expiries_arr <= 0.0):
        raise ValueError("all expiries must be positive")

    rng = np.random.default_rng(seed)
    p_true = (true_heston or DEFAULT_TRUE_HESTON) if mode == "heston" else None
    svi_true = default_svi_slices(expiries_arr) if mode == "svi" else None

    rows = []
    n_grid = int(round((moneyness_hi - moneyness_lo) / moneyness_step)) + 1
    base_grid = np.linspace(moneyness_lo, moneyness_hi, n_grid)

    for T in expiries_arr:
        F = spot * np.exp((rate - div_yield) * T)
        if mode == "heston":
            atm_vol = np.sqrt(max(p_true.v0, 0.5 * (p_true.v0 + p_true.theta)))
        else:
            atm_vol = float(np.sqrt(svi_total_variance(0.0, svi_true[float(T)]) / T))
        width = 3.3 * atm_vol * np.sqrt(T)
        mny = base_grid[(np.log(base_grid) >= -width) & (np.log(base_grid) <= width)]
        if mny.size < 5:  # always keep at least 5 quotes around ATM
            mny = np.exp(np.linspace(-width, width, 5))
        strikes = np.round(F * mny, 4)
        k = np.log(strikes / F)

        if mode == "heston":
            mids = np.asarray(heston_call_gl(spot, strikes, float(T), rate, div_yield, p_true))
        else:
            w = np.asarray(svi_total_variance(k, svi_true[float(T)]))
            vols = np.sqrt(w / T)
            mids = np.array(
                [bs_price(spot, kk, float(T), rate, div_yield, v, "call") for kk, v in zip(strikes, vols)]
            )

        if bid_ask:
            # Half-spread widens in the wings (quotes are worth less there).
            hs = half_spread_atm * (1.0 + 4.0 * np.abs(k)) * np.maximum(mids, 1e-4)
            jitter = rng.uniform(-0.5, 0.5, size=mids.size) * hs
            mid_noisy = np.maximum(mids + jitter, 1e-8)
            bid = np.maximum(mid_noisy - hs, 0.0)
            ask = mid_noisy + hs
        else:
            mid_noisy, bid, ask = mids, None, None

        for j in range(strikes.size):
            row = {
                "expiry": float(T),
                "strike": float(strikes[j]),
                "forward": float(F),
                "moneyness": float(strikes[j] / F),
                "log_moneyness": float(k[j]),
                "call_mid": float(mid_noisy[j]),
            }
            if bid_ask:
                row["call_bid"] = float(bid[j])
                row["call_ask"] = float(ask[j])
            rows.append(row)

    df = pd.DataFrame(rows)
    return ChainData(
        df=df,
        spot=spot,
        rate=rate,
        div_yield=div_yield,
        mode=mode,
        true_heston=p_true,
        true_svi=svi_true,
    )
