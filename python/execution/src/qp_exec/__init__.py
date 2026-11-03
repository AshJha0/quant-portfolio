"""qp_exec: order book, order flow, fills, impact, Hawkes models and schedules.

Subpackages
-----------
lob
    The reference limit order book (``ReferenceBook``) and its semantics
    (``lob/SPEC.md``), order-flow generators, markouts, impact fitting and
    stylised-fact checks. Other lanes build on ``qp_exec.lob`` only through
    the names exported from ``qp_exec.lob.__init__``.
hawkes
    Exponential-kernel Hawkes processes: simulation, maximum likelihood and
    time-rescaling goodness of fit.
schedules
    TWAP, VWAP and POV on a volume forecast, optimal schedules under transient
    impact (Obizhaeva-Wang, Gatheral), and the passive-or-cross decision.
"""

__version__ = "3.0.0.dev0"
