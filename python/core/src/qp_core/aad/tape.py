"""A tape for reverse-mode (adjoint) algorithmic differentiation.

Plain idea. Every arithmetic step of a pricer is written down, in order, on
a tape: the result, which earlier results it used, and the local partial
derivative of the result with respect to each of them. To get the gradient
of the final price with respect to every input, walk the tape once from the
end back to the start, pushing each node's adjoint ("how much the price
moves per unit of this node") into its parents. One backward walk gives all
first-order sensitivities, whatever their number.

Formal statement (Griewank and Walther 2008, *Evaluating Derivatives*,
ch. 3; Savine 2018, *Modern Computational Finance*, ch. 9). For nodes
``v_i = phi_i(v_j : j in parents(i))`` the adjoints ``vbar_i = dy/dv_i``
satisfy ``vbar_j = sum_{i : j in parents(i)} vbar_i * d phi_i / d v_j``, seeded
with ``ybar = 1`` and evaluated in reverse order. The cost is a small
constant multiple of one evaluation (Griewank's "cheap gradient
principle": at most about 4 to 5 times for the arithmetic, plus the
bookkeeping of recording the tape).

Worked example. ``y = x0 * x1 + exp(x0)`` at ``x0 = 1, x1 = 2``:

>>> tape = Tape()
>>> x0, x1 = tape.var(1.0), tape.var(2.0)
>>> y = x0 * x1 + exp(x0)
>>> round(y.value, 6)
4.718282
>>> g = tape.gradient(y, [x0, x1])
>>> [round(v, 6) for v in g]       # dy/dx0 = x1 + e, dy/dx1 = x0
[4.718282, 1.0]

Design. Nodes have at most two parents, stored as one flat list of
``(parent, partial, parent, partial)`` tuples, which is the layout of
Savine's ``Node``. ``mark()`` / ``rewind(mark)`` keep the inputs and drop
everything recorded after them; the Monte Carlo pricer uses that to record,
sweep and forget one path at a time while the adjoints of the inputs
accumulate across paths (Savine 2018, s. 10.3). Pricers are written once
against the helper functions in this module (``exp``, ``log``, ``sqrt``,
``ncdf``), which take plain floats as well, so the same code runs as the
plain-float pricer whose time is the denominator of the cost ratio.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_NONE = -1
_LEAF = (_NONE, 0.0, _NONE, 0.0)


class Tape:
    """Recording of one computation; see the module docstring."""

    __slots__ = ("_adj", "_nodes")

    def __init__(self) -> None:
        # One tuple ``(a, da, b, db)`` per node: one append per operation,
        # where four parallel lists cost four (the dominant cost of a
        # small pricer under CPython).
        self._nodes: list[tuple[int, float, int, float]] = []
        self._adj: list[float] = []

    def __len__(self) -> int:
        return len(self._nodes)

    def _push(self, value: float, a: int, da: float, b: int, db: float) -> Var:
        nodes = self._nodes
        nodes.append((a, da, b, db))
        return Var(self, len(nodes) - 1, value)

    def var(self, value: float) -> Var:
        """A new independent input (a leaf node)."""
        return self._push(float(value), _NONE, 0.0, _NONE, 0.0)

    def vars(self, values: Sequence[float]) -> list[Var]:
        """Several independent inputs at once (one ``extend``)."""
        first = len(self._nodes)
        self._nodes.extend([_LEAF] * len(values))
        return [Var(self, first + i, float(v)) for i, v in enumerate(values)]

    def clear(self) -> None:
        """Forget every node but keep the list objects, so a tape reused
        across calls (``bs_greeks_aad``) skips building a new one."""
        self.rewind(0)

    def mark(self) -> int:
        """Position to come back to with :meth:`rewind`."""
        return len(self._nodes)

    def rewind(self, mark: int) -> None:
        """Forget every node recorded after ``mark``; adjoints before it stay."""
        del self._nodes[mark:]
        if len(self._adj) > mark:
            del self._adj[mark:]

    def zero_adjoints(self) -> None:
        self._adj = [0.0] * len(self._nodes)

    def backward(self, output: Var, seed: float = 1.0, stop: int = 0) -> None:
        """Add ``seed * d output / d node`` into every node's adjoint.

        Walks from ``output`` down to node ``stop``; nodes below ``stop``
        receive their contributions but are not propagated further (they are
        inputs). Adjoints accumulate across calls until :meth:`zero_adjoints`.
        """
        if output.tape is not self:
            raise ValueError("output was recorded on a different tape")
        adj = self._adj
        n = len(self._nodes)
        if len(adj) < n:
            adj.extend([0.0] * (n - len(adj)))
        adj[output.idx] += seed
        self.propagate(output.idx + 1, stop)

    def propagate(self, start: int, stop: int = 0) -> None:
        """Push adjoints of nodes ``start-1`` down to ``stop`` into their parents.

        After a per-path Monte Carlo loop has accumulated adjoints on nodes
        recorded before the paths (``backward(..., stop=mark)``), one
        ``propagate(mark)`` carries them back to the inputs.
        """
        adj = self._adj
        nodes = self._nodes
        if len(adj) < len(nodes):
            adj.extend([0.0] * (len(nodes) - len(adj)))
        for i in range(start - 1, stop - 1, -1):
            w = adj[i]
            if w == 0.0:
                continue
            a, da, b, db = nodes[i]
            if a >= 0:
                adj[a] += w * da
                if b >= 0:
                    adj[b] += w * db

    def adjoints(self, vs: Sequence[Var]) -> list[float]:
        adj = self._adj
        n = len(adj)
        return [adj[v.idx] if v.idx < n else 0.0 for v in vs]

    def adjoint(self, v: Var) -> float:
        return self._adj[v.idx] if v.idx < len(self._adj) else 0.0

    def gradient(self, output: Var, inputs: Sequence[Var]) -> list[float]:
        """``[d output / d x for x in inputs]`` in one reverse sweep."""
        self.zero_adjoints()
        self.backward(output)
        return [self.adjoint(x) for x in inputs]


def _val(x: Real) -> float:
    return x.value if isinstance(x, Var) else float(x)


class Var:
    """A value recorded on a :class:`Tape`."""

    __slots__ = ("idx", "tape", "value")

    def __init__(self, tape: Tape, idx: int, value: float) -> None:
        self.tape = tape
        self.idx = idx
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Var({self.value!r})"

    # Comparisons look at values only; branches are not differentiated.
    def __lt__(self, o: Real) -> bool:
        return self.value < _val(o)

    def __gt__(self, o: Real) -> bool:
        return self.value > _val(o)

    def __le__(self, o: Real) -> bool:
        return self.value <= _val(o)

    def __ge__(self, o: Real) -> bool:
        return self.value >= _val(o)

    def __float__(self) -> float:
        return self.value

    def __add__(self, o: Real) -> Var:
        if isinstance(o, Var):
            return self.tape._push(self.value + o.value, self.idx, 1.0, o.idx, 1.0)
        return self.tape._push(self.value + o, self.idx, 1.0, _NONE, 0.0)

    __radd__ = __add__

    def __sub__(self, o: Real) -> Var:
        if isinstance(o, Var):
            return self.tape._push(self.value - o.value, self.idx, 1.0, o.idx, -1.0)
        return self.tape._push(self.value - o, self.idx, 1.0, _NONE, 0.0)

    def __rsub__(self, o: float) -> Var:
        return self.tape._push(o - self.value, self.idx, -1.0, _NONE, 0.0)

    def __mul__(self, o: Real) -> Var:
        if isinstance(o, Var):
            return self.tape._push(self.value * o.value, self.idx, o.value, o.idx, self.value)
        if o == 1.0:
            # ``x * 1`` is ``x`` bit for bit: record nothing (calls, phi = 1).
            return self
        return self.tape._push(self.value * o, self.idx, float(o), _NONE, 0.0)

    __rmul__ = __mul__

    def __truediv__(self, o: Real) -> Var:
        if isinstance(o, Var):
            q = self.value / o.value
            return self.tape._push(q, self.idx, 1.0 / o.value, o.idx, -q / o.value)
        return self.tape._push(self.value / o, self.idx, 1.0 / o, _NONE, 0.0)

    def __rtruediv__(self, o: float) -> Var:
        q = o / self.value
        return self.tape._push(q, self.idx, -q / self.value, _NONE, 0.0)

    def __neg__(self) -> Var:
        return self.tape._push(-self.value, self.idx, -1.0, _NONE, 0.0)


Real = float | Var


def exp(x: Real) -> Real:
    if isinstance(x, Var):
        e = math.exp(x.value)
        return x.tape._push(e, x.idx, e, _NONE, 0.0)
    return math.exp(x)


def log(x: Real) -> Real:
    if isinstance(x, Var):
        return x.tape._push(math.log(x.value), x.idx, 1.0 / x.value, _NONE, 0.0)
    return math.log(x)


def sqrt(x: Real) -> Real:
    if isinstance(x, Var):
        s = math.sqrt(x.value)
        return x.tape._push(s, x.idx, 0.5 / s, _NONE, 0.0)
    return math.sqrt(x)


def ncdf(x: Real) -> Real:
    """Standard normal CDF ``0.5 erfc(-x/sqrt 2)``; derivative is the density."""
    if isinstance(x, Var):
        v = x.value
        return x.tape._push(0.5 * math.erfc(-v / _SQRT2), x.idx,
                            _INV_SQRT_2PI * math.exp(-0.5 * v * v), _NONE, 0.0)
    return 0.5 * math.erfc(-x / _SQRT2)


# Fused operations. Each float branch is the plain composition, so the float
# pricer computes exactly the bits it did unfused; a Var records one node
# with its local partials preaccumulated (statement-level preaccumulation:
# Griewank and Walther 2008, s. 10.2; Savine 2018, ch. 15). Fewer nodes means
# fewer Python objects and a shorter reverse sweep. The Rust port has the
# same four as default methods of ``Real``.


def exp_neg_mul(x: Real, y: Real) -> Real:
    """``exp(-x * y)``, e.g. a discount factor ``exp(-r T)``."""
    if not isinstance(x, Var) and not isinstance(y, Var):
        return math.exp(-x * y)  # the plain-float pricer: checked first
    if isinstance(x, Var) and isinstance(y, Var):
        e = math.exp(-x.value * y.value)
        return x.tape._push(e, x.idx, -y.value * e, y.idx, -x.value * e)
    return exp(-x * y)


def ln_div(x: Real, y: Real) -> Real:
    """``log(x / y)``, e.g. the log-moneyness ``log(S / K)``."""
    if not isinstance(x, Var) and not isinstance(y, Var):
        return math.log(x / y)
    if isinstance(x, Var) and isinstance(y, Var):
        return x.tape._push(math.log(x.value / y.value), x.idx, 1.0 / x.value,
                            y.idx, -1.0 / y.value)
    return log(x / y)


def ncdf_scaled(x: Real, c: float) -> Real:
    """``ncdf(c * x)`` for a constant ``c``."""
    if isinstance(x, Var):
        v = c * x.value
        return x.tape._push(0.5 * math.erfc(-v / _SQRT2), x.idx,
                            c * _INV_SQRT_2PI * math.exp(-0.5 * v * v), _NONE, 0.0)
    return 0.5 * math.erfc(-(c * x) / _SQRT2)


def half_square(x: Real) -> Real:
    """``0.5 * x * x``."""
    if isinstance(x, Var):
        v = x.value
        return x.tape._push(0.5 * v * v, x.idx, v, _NONE, 0.0)
    return 0.5 * x * x


def value(x: Real) -> float:
    """The float value of a plain number or a :class:`Var`."""
    return _val(x)
