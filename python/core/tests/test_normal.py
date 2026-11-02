"""qp_core.normal against scipy and the lower-tail limit."""
import math

import pytest
from qp_core.normal import norm_cdf, norm_pdf
from scipy.stats import norm


@pytest.mark.parametrize("x", [-30.0, -20.0, -8.0, -1.5, 0.0, 0.3, 2.0, 9.0])
def test_norm_cdf_matches_scipy(x):
    # scipy's ndtr and libm erfc differ by a few ulp-scale relative errors in the far tail.
    assert norm_cdf(x) == pytest.approx(norm.cdf(x), rel=1e-12, abs=0.0)


@pytest.mark.parametrize("x", [-10.0, -1.0, 0.0, 0.5, 4.0])
def test_norm_pdf_matches_scipy(x):
    assert norm_pdf(x) == pytest.approx(norm.pdf(x), rel=1e-13)


def test_lower_tail_keeps_relative_precision():
    # 0.5 (1 + erf(x/sqrt 2)) underflows to 0 here; erfc does not.
    assert 0.5 * (1.0 + math.erf(-30.0 / math.sqrt(2.0))) == 0.0
    assert math.isclose(norm_cdf(-30.0), 4.906713927148187e-198, rel_tol=1e-12)


def test_option_packages_use_the_shared_definition():
    eq_iv = pytest.importorskip("eq_options._iv_core")
    fx_iv = pytest.importorskip("fx_options._iv_core")
    assert eq_iv.norm_cdf is norm_cdf
    assert fx_iv.norm_cdf is norm_cdf
