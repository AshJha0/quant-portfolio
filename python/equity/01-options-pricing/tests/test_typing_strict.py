"""W22: example user code type-checks under ``mypy --strict``.

The snippet uses the scalar and array forms of ``bs_price`` and
``gk_price``, the typed results and the ``engine=`` wrappers, and asserts
the inferred types with ``assert_type`` so a wrong overload fails the check.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

USER_CODE = textwrap.dedent('''
    from typing import assert_type

    import numpy as np
    from numpy.typing import NDArray

    from eq_options import BSGreeks, bs_greeks, bs_price
    from eq_options.engine import bs_price_batch
    from eq_var import MonteCarloRisk
    from eq_var.engine import monte_carlo_var
    from fx_options.garman_kohlhagen import gk_price

    p = bs_price(100.0, 100.0, 1.0, 0.05, 0.2)
    assert_type(p, float)
    strikes = np.linspace(80.0, 120.0, 5)
    arr = bs_price(np.full(5, 100.0), strikes, 1.0, 0.05, 0.2)
    assert_type(arr, NDArray[np.float64])
    g = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2)
    assert_type(g, BSGreeks)
    fx = gk_price(1.1, 1.1, 0.5, 0.03, 0.01, 0.1, "call")
    assert_type(fx, float)
    b = bs_price_batch(strikes, strikes, np.ones(5), np.zeros(5), np.full(5, 0.2),
                       np.zeros(5), np.ones(5, dtype=bool))
    assert_type(b, NDArray[np.float64])
    v = monte_carlo_var(np.array([1.0e6]), np.array([[1.0e-4]]), 0.01, 1000)
    assert_type(v, float)

    def summary(r: MonteCarloRisk) -> float:
        return r.var + r.es
''')


def test_user_code_passes_mypy_strict(tmp_path, monkeypatch):
    api = pytest.importorskip("mypy.api")
    # Point mypy at the source roots the test process imports from, so the
    # check works for editable installs and PYTHONPATH runs alike.
    roots = []
    for name in ("qp_core", "eq_options", "eq_var", "fx_options"):
        mod = pytest.importorskip(name)
        root = str(Path(mod.__file__).resolve().parent.parent)
        if root not in roots:
            roots.append(root)
    monkeypatch.setenv("MYPYPATH", os.pathsep.join(roots))
    f = tmp_path / "user_code.py"
    f.write_text(USER_CODE, encoding="utf-8")
    out, err, status = api.run(["--strict", "--no-incremental", "--cache-dir", str(tmp_path / "c"),
                                "--python-version", "3.12", str(f)])
    assert status == 0, out + err
