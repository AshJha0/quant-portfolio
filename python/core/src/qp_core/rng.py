"""One seed-to-generator rule for the Python engines (plan W20).

Plain idea
----------
Every Monte Carlo routine in the repository takes either a seed or a NumPy
``Generator``. Before Round 8 each package had its own five-line helper for
that (``eq_options.monte_carlo._make_rng``, ``eq_var.monte_carlo_var._as_rng``,
``fx_options.monte_carlo._resolve_rng``, ``fx_var.monte_carlo_var._as_rng``,
``fx_surface.heston_mc._rng``, ``eq_pairs.data.synthetic.make_rng`` and
others). They agree on the stream, ``numpy.random.default_rng(seed)``, but not
on what counts as a valid seed: one rejects negative integers with a
``ValueError``, the others pass the value to NumPy, which raises a
``ValueError`` for a negative integer and a ``TypeError`` for a float, and
silently accepts ``True`` as the seed 1. This module is the one rule.

Formal statement
----------------
``as_generator(seed)`` returns

* ``seed`` itself when it is a ``numpy.random.Generator`` (the caller keeps
  control of the stream);
* ``numpy.random.default_rng(seed)`` (PCG64, O'Neill 2014, seeded through
  ``SeedSequence``) when ``seed`` is ``None`` or a non-negative ``int``,
  including NumPy integers;
* a ``ValueError`` for anything else, ``bool`` included.

So for any valid integer seed the draws are bit-identical to the helpers it
replaces. ``spawn_generators(seed, n)`` gives ``n`` independent child streams
through ``SeedSequence.spawn`` (NumPy's documented way to get non-overlapping
parallel streams), for code that splits work across processes.

Worked example
--------------
>>> g = as_generator(7)
>>> float(g.standard_normal()) == float(np.random.default_rng(7).standard_normal())
True
>>> as_generator(g) is g
True
>>> a, b = spawn_generators(7, 2)
>>> float(a.random()) != float(b.random())
True
"""

from __future__ import annotations

import numbers

import numpy as np

__all__ = ["SeedLike", "as_generator", "spawn_generators"]

SeedLike = int | np.integer | np.random.Generator | None


def _check_seed(seed: object) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f"seed must be None, a non-negative int or a numpy Generator, got {seed!r}")
    value = int(seed)
    if value < 0:
        raise ValueError(f"seed must be non-negative, got {value}")
    return value


def as_generator(seed: SeedLike = None) -> np.random.Generator:
    """A NumPy ``Generator`` from a seed, or the given ``Generator`` unchanged."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(_check_seed(seed))


def spawn_generators(seed: int | np.integer | None, n: int) -> list[np.random.Generator]:
    """``n`` independent generators spawned from one seed (``SeedSequence.spawn``)."""
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    children = np.random.SeedSequence(_check_seed(seed)).spawn(int(n))
    return [np.random.Generator(np.random.PCG64(c)) for c in children]
