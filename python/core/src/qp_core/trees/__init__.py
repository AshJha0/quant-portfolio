"""Shared lattice kernels (W20).

One implementation per method, called by the equity and FX wrappers. An FX
option is an equity option with the foreign rate ``r_f`` in the place of
the dividend yield ``q`` (Garman and Kohlhagen 1983), so the kernels take
``(r, q)`` and the FX wrappers pass ``(r_d, r_f)``.
"""

from qp_core.trees.crr import crr_lattice

__all__ = ["crr_lattice"]
