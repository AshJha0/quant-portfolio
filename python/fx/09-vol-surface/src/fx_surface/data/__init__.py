"""Synthetic FX market data (offline, seeded). No live loaders here:
this project is convention/model focused, so all tests and examples run
on the deterministic generators in :mod:`fx_surface.data.synthetic`."""

from .synthetic import (
    FXMarketData,
    MarketSlice,
    calibration_slices,
    em_high_vol_market,
    eurusd_market,
    market_from_heston,
    usdjpy_market,
)

__all__ = [
    "FXMarketData",
    "MarketSlice",
    "calibration_slices",
    "em_high_vol_market",
    "eurusd_market",
    "market_from_heston",
    "usdjpy_market",
]
