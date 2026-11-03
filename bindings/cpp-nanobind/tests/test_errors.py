"""Engine failures raise the qp_core.errors classes (W8).

Bad arguments raise ``InputError`` (a ``ValueError``, as before); a numerical
method failing on well-formed input raises ``ConvergenceError`` (a
``RuntimeError`` and, for compatibility, a ``ValueError``). The covariance case uses a materially indefinite matrix,
which the engines refuse after the bounded Cholesky jitter rather than
repair silently.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

errors = pytest.importorskip("qp_core.errors")
qp = importlib.import_module("qp_cpp")

INDEFINITE = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1


@pytest.mark.parametrize(
    "call",
    [
        lambda: qp.eq_bs_price(-1.0, 100.0, 1.0, 0.0, 0.2),
        lambda: qp.eq_bs_price(100.0, 100.0, 1.0, 0.0, 0.2, option_type="straddle"),
        lambda: qp.fx_gk_price(1.1, 1.1, 1.0, 0.03, 0.01, -0.1, "call"),
        lambda: qp.eq_implied_vol(-5.0, 100.0, 100.0, 1.0, 0.0, 0.0, "call"),
        lambda: qp.eq_bs_price_batch(np.ones(3), np.ones(2), np.ones(3), np.zeros(3), np.ones(3), np.zeros(3), np.ones(3, dtype=bool)),
        lambda: qp.eq_historical_var(np.array([])),
    ],
    ids=["negative-spot", "bad-option-type", "negative-vol", "iv-below-bound", "length-mismatch", "empty-pnl"],
)
def test_bad_arguments_raise_input_error(call):
    with pytest.raises(errors.InputError) as info:
        call()
    assert isinstance(info.value, ValueError)
    assert isinstance(info.value, errors.QPError)


def test_batch_error_keeps_the_element_index_and_class():
    s = np.array([100.0, 100.0, -1.0])
    one = np.ones(3)
    with pytest.raises(errors.InputError, match="element 2"):
        qp.eq_bs_price_batch(s, 100.0 * one, one, 0.0 * one, 0.2 * one, 0.0 * one, np.ones(3, dtype=bool))


def test_indefinite_covariance_raises_convergence_error():
    with pytest.raises(errors.ConvergenceError) as info:
        qp.eq_monte_carlo_var(np.array([1.0, 1.0]), INDEFINITE, n_paths=1000)
    assert isinstance(info.value, RuntimeError)
    assert isinstance(info.value, ValueError)  # old except clauses still catch it


def test_qp_core_error_classes_are_the_same_objects():
    try:
        qp.eq_bs_price(-1.0, 100.0, 1.0, 0.0, 0.2)
    except errors.QPError as exc:
        assert type(exc) is errors.InputError
    else:  # pragma: no cover
        pytest.fail("no error raised")
