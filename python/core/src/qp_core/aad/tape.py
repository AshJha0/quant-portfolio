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

Design. Nodes have at most two parents, stored in four flat lists (parent
indices and partials), which is the layout of Savine's ``Node`` without the
per-node allocation. ``mark()`` / ``rewind(mark)`` keep the inputs and drop
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


class Tape:
    """Recording of one computation; see the module docstring."""

    __slots__ = ("_a", "_adj", "_b", "_da", "_db")

    def __init__(self) -> None:
        self._a: list[int] = []
        self._b: list[int] = []
        self._da: list[float] = []
        self._db: list[float] = []
        self._adj: list[float] = []

    def __len__(self) -> int:
        return len(self._a)

    def _push(self, value: float, a: int, da: float, b: int, db: float) -> Var:
        self._a.append(a)
        self._da.append(da)
        self._b.append(b)
        self._db.append(db)
        return Var(self, len(self._a) - 1, value)

    def var(self, value: float) -> Var:
        """A new independent input (a leaf node)."""
        return self._push(float(value), _NONE, 0.0, _NONE, 0.0)

    def mark(self) -> int:
        """Position to come back to with :meth:`rewind`."""
        return len(self._a)

    def rewind(self, mark: int) -> None:
        """Forget every node recorded after ``mark``; adjoints before it stay."""
        del self._a[mark:], self._b[mark:], self._da[mark:], self._db[mark:]
        if len(self._adj) > mark:
            del self._adj[mark:]

    def zero_adjoints(self) -> None:
        self._adj = [0.0] * len(self._a)

    def backward(self, output: Var, seed: float = 1.0, stop: int = 0) -> None:
        """Add ``seed * d output / d node`` into every node's adjoint.

        Walks from ``output`` down to node ``stop``; nodes below ``stop``
        receive their contributions but are not propagated further (they are
        inputs). Adjoints accumulate across calls until :meth:`zero_adjoints`.
        """
        if output.tape is not self:
            raise ValueError("output was recorded on a different tape")
        adj = self._adj
        n = len(self._a)
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
        if len(adj) < len(self._a):
            adj.extend([0.0] * (len(self._a) - len(adj)))
        a_, b_, da_, db_ = self._a, self._b, self._da, self._db
        for i in range(start - 1, stop - 1, -1):
            w = adj[i]
            if w == 0.0:
                continue
            a = a_[i]
            if a >= 0:
                adj[a] += w * da_[i]
                b = b_[i]
                if b >= 0:
                    adj[b] += w * db_[i]

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


def value(x: Real) -> float:
    """The float value of a plain number or a :class:`Var`."""
    return _val(x)
