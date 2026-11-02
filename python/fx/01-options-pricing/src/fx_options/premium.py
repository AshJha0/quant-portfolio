"""FX option premium quote styles and conversions.

Plain idea: one FX option price can be quoted four ways, depending on which
currency the premium is paid in and per unit of which notional. For EURUSD
(foreign EUR, domestic USD) the Garman-Kohlhagen value is USD per 1 EUR of
notional: "domestic pips". Divide by spot and it is EUR per 1 EUR, a
percentage of the EUR notional ("foreign %"). Divide by strike and it is
USD per 1 USD of the USD notional ("domestic %"). Divide by both and it is
EUR per 1 USD ("foreign pips").

Formal statement (Wystup 2017, "FX Options and Structured Products", 2nd
ed., Wiley, section 1.5.6; Clark 2011, ch. 2). With ``V`` the GK value in
domestic per unit of foreign notional, spot ``S`` and strike ``K``:

====================  ===============  ================================
style                 value            meaning
====================  ===============  ================================
``"dom_pips"``        ``V``            domestic per 1 foreign
``"for_pct"``         ``V / S``        foreign per 1 foreign
``"dom_pct"``         ``V / K``        domestic per 1 domestic
``"for_pips"``        ``V / (S K)``    foreign per 1 domestic
====================  ===============  ================================

Foreign pips is the domestic-pips price of the mirror option in the
inverted pair: a FOR/DOM call at ``K`` is a DOM/FOR put at ``1/K`` with the
rates swapped, ``V_call(S, K, r_d, r_f) / (S K) = V_put(1/S, 1/K, r_f, r_d)``
(the test oracle).

Worked example: ``V = 0.0250`` USD per EUR, ``S = 1.10``, ``K = 1.12``:
2.2727 % of EUR notional, 2.2321 % of USD notional, 0.020292 EUR per USD.
"""

from __future__ import annotations

import math

__all__ = ["PREMIUM_STYLES", "convert_premium", "premium_amount"]

PREMIUM_STYLES = ("dom_pips", "for_pct", "dom_pct", "for_pips")


def _scale(style: str, S: float, K: float) -> float:
    """Factor turning a domestic-pips value into ``style``."""
    if style not in PREMIUM_STYLES:
        raise ValueError(f"style must be one of {PREMIUM_STYLES}, got {style!r}")
    return {"dom_pips": 1.0, "for_pct": 1.0 / S, "dom_pct": 1.0 / K, "for_pips": 1.0 / (S * K)}[style]


def convert_premium(value: float, from_style: str, to_style: str, S: float, K: float) -> float:
    """Convert a premium quote between styles (``value`` per unit notional)."""
    for name, v in (("value", value), ("S", S), ("K", K)):
        if not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
    if S <= 0.0 or K <= 0.0:
        raise ValueError("S and K must be > 0")
    return value / _scale(from_style, S, K) * _scale(to_style, S, K)


def premium_amount(dom_pips: float, foreign_notional: float, S: float, K: float,
                   pay_in: str = "domestic") -> float:
    """Cash premium for a foreign notional, paid in ``"domestic"`` or ``"foreign"``."""
    if pay_in not in ("domestic", "foreign"):
        raise ValueError(f"pay_in must be 'domestic' or 'foreign', got {pay_in!r}")
    amount = dom_pips * foreign_notional
    return amount if pay_in == "domestic" else convert_premium(amount, "dom_pips", "for_pct", S, K)
