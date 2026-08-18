"""Market data: deterministic synthetic generators (offline, seeded) and
import-guarded live loaders (network opt-in via FX_RATES_ALLOW_NETWORK=1)."""

from .synthetic import (
    REGIMES,
    SyntheticMarket,
    build_market_state,
    generate_market_quotes,
    sample_book,
    third_currency_curve,
)

__all__ = [
    "REGIMES",
    "SyntheticMarket",
    "build_market_state",
    "generate_market_quotes",
    "sample_book",
    "third_currency_curve",
]
