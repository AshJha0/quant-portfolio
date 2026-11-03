"""Exotic options: barriers, touches, double barriers, vanna-volga barrier
adjustment, Asians, Longstaff-Schwartz Americans, TARFs and particle-method
local stochastic volatility.

Each module states its plain idea, the formal result with its source, and a
worked example. Closed forms are checked against published tables (Haug
2007) or an independent Monte Carlo; the closed-form pricers have a Rust
twin in ``rust/core/src/exotics.rs`` held to 1e-12 by golden vectors.
"""
