# Python: knowledge check (questions only)

Two parts. Part A: fourteen "what does this print?" snippets on the parts of
the language that cause real bugs in research code. Part B: six short coding
tasks; reference solutions are in [`solutions.py`](solutions.py). Answers
for part A are in [`keys/answers.md`](keys/answers.md).

How it is checked: `career-prep/tests/test_python_pack.py` runs every snippet
below and compares its output with the key, and tests each part-B solution
against a brute-force version on random inputs. Outputs are written with
" | " between lines. CPython 3.12 or later is assumed.

References: the Python Language Reference (sections 4.2 on naming and
binding, 8.4 on `try`); McKinney, *Python for Data Analysis* (3rd ed.); the
pandas user guide on time series.

## Part A: what does this print?

### Y01 Mutable default argument
```python
def add(x, acc=[]):
    acc.append(x)
    return acc
add(1)
print(add(2))
```

### Y02 Late-binding closures
```python
fs = [lambda: i for i in range(3)]
print([f() for f in fs])
```

### Y03 Aliased nested lists
```python
grid = [[0] * 3] * 2
grid[0][0] = 1
print(grid)
```

### Y04 Float equality
```python
print(0.1 + 0.2 == 0.3, abs(0.1 + 0.2 - 0.3) < 1e-12)
```

### Y05 Generators are single-use
```python
g = (x * x for x in range(4))
print(sum(g), sum(g))
```

### Y06 Local scope
```python
n = 0
def bump():
    try:
        n += 1
    except UnboundLocalError:
        return "unbound"
print(bump())
```

### Y07 return in finally
```python
def f():
    try:
        return 1
    finally:
        return 2
print(f())
```

### Y08 Banker's rounding
```python
print(round(0.5), round(1.5), round(2.5))
```

### Y09 Integer division and modulo with negatives
```python
print(-7 // 2, -7 % 2)
```

### Y10 Shallow copy
```python
import copy
a = {"k": [1]}
b = copy.copy(a)
b["k"].append(2)
print(a)
```

### Y11 NumPy broadcasting
```python
import numpy as np
print((np.ones((3, 1)) + np.arange(4)).shape)
```

### Y12 NumPy views
```python
import numpy as np
a = np.arange(5)
b = a[1:3]
b[0] = 99
print(a.tolist())
```

### Y13 pandas shift: what is known when
```python
import pandas as pd
px = pd.Series([100.0, 101.0, 99.0, 102.0])
signal = px.pct_change()
print(signal.shift(1).round(4).tolist())
```

### Y14 pandas resample label
```python
import pandas as pd
idx = pd.date_range("2026-01-05 09:00", periods=4, freq="30min")
s = pd.Series([1, 2, 3, 4], index=idx)
print(s.resample("1h").sum().tolist())
```

## Part B: write the function

- **YB1** `max_drawdown(prices)`: largest peak-to-trough fall as a fraction of the peak, in O(n).
- **YB2** `vwap(trades)`: volume-weighted average price from `(price, qty)` pairs; raise on zero volume.
- **YB3** `rolling_max(xs, k)`: the maximum of each window of length `k`, in O(n) with a deque.
- **YB4** `LRUCache(capacity)` with `get` and `put` in O(1).
- **YB5** `merge_sorted(streams)`: merge already-sorted iterables of `(timestamp, value)` into one sorted list, lazily.
- **YB6** `asof_join(trades, quotes)`: for each trade time, the last quote at or before it (pandas `merge_asof`).

## Discussion (not graded)

- **YD1** What does the GIL stop you doing, and what are the three usual ways around it?
- **YD2** Why is `df.apply(row_function, axis=1)` slow, and what do you write instead?
- **YD3** How do you make a research result reproducible a year later?
