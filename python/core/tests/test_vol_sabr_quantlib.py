"""qp_core.vol.sabr against QuantLib 1.44 (``sabrVolatility``, ``shiftedSabrVolatility``).

Both implement Hagan et al. (2002) eq. (2.17a), so the ``hagan`` variant must
agree to rounding. Skips without QuantLib; the CI ``quantlib`` job forbids skips.
"""

from __future__ import annotations

import pytest
from qp_core.vol.sabr import sabr_vol

ql = pytest.importorskip("QuantLib")

ARGS = (0.03, 5.0, 0.02, 0.5, -0.3, 0.4)


@pytest.mark.parametrize("beta", [0.0, 0.3, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("k", [0.005, 0.015, 0.03, 0.045, 0.08])
def test_hagan_variant_matches_quantlib(beta, k):
    F, T, _, _, rho, nu = ARGS
    a = 0.02 * (F / 0.03) ** (beta - 0.5)  # comparable ATM vol across beta
    ours = sabr_vol(F, k, T, a, beta, rho, nu, variant="hagan")
    theirs = ql.sabrVolatility(k, F, T, a, beta, nu, rho)
    assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-14)


def test_shifted_matches_quantlib_shifted():
    F, K, T, s = -0.002, 0.001, 2.0, 0.03
    ours = sabr_vol(F, K, T, 0.01, 0.5, -0.2, 0.3, shift=s, variant="hagan")
    theirs = ql.shiftedSabrVolatility(K, F, T, 0.01, 0.5, 0.3, -0.2, s)
    assert ours == pytest.approx(theirs, rel=1e-12)
