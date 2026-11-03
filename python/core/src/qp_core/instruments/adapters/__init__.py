"""Adapters from the per-package trade and market classes to the trade layer.

Only this sub-package may import product packages, and only lazily, so
``qp_core`` itself depends on none of them. Each adapter states which old
class it converts and is tested to reproduce the old ``value`` to 1e-12.

=========================  ==================================================
``fx_rates_adapter``       ``fx_rates.MarketState`` -> ``MarketSnapshot``;
                           ``fx_rates.FXForward`` / ``FXSwap`` -> native
=========================  ==================================================

Still to adapt (integration list, plan W10): ``fx_pairs/signals.py`` trades,
``fx_credit/saccr.py`` and ``settlement.py`` trades, ``eq_var/portfolio.py``
positions. Those lanes call :func:`qp_core.instruments.register` for any
product the native set does not cover.
"""

from __future__ import annotations

__all__: list[str] = []
