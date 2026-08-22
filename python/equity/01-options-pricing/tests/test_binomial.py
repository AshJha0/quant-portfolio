"""CRR binomial tree: convergence to BS, American exercise properties."""

import math

import numpy as np
import pytest

from eq_options import bs_price, crr_price, early_exercise_premium

CASES = [
    # (S, K, T, r, sigma, q)
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),
    (100.0, 110.0, 0.5, 0.03, 0.30, 0.02),
    (100.0, 80.0, 2.0, 0.01, 0.15, 0.04),
    (50.0, 60.0, 0.25, -0.01, 0.40, 0.00),
]


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
@pytest.mark.parametrize("otype", ["call", "put"])
def test_european_tree_converges_to_bs_500_steps(
    S: float, K: float, T: float, r: float, sigma: float, q: float, otype: str
) -> None:
    bs = bs_price(S, K, T, r, sigma, q, otype)
    tree = crr_price(S, K, T, r, sigma, q, otype, "european", 500)
    assert tree == pytest.approx(bs, abs=2e-2, rel=2e-3)


def test_european_tree_tight_at_5000_steps() -> None:
    bs = bs_price(100, 100, 1, 0.05, 0.2, 0.02, "call")
    tree = crr_price(100, 100, 1, 0.05, 0.2, 0.02, "call", "european", 5000)
    assert tree == pytest.approx(bs, abs=2e-3)


def test_monotone_convergence_sanity() -> None:
    """|error| decreases as steps double (same parity kills CRR odd/even
    oscillation); error * n stays bounded (O(1/n) rate)."""
    S, K, T, r, sigma, q = 100.0, 105.0, 1.0, 0.04, 0.25, 0.01
    bs = bs_price(S, K, T, r, sigma, q, "call")
    steps = [50, 100, 200, 400, 800]
    errs = [
        abs(crr_price(S, K, T, r, sigma, q, "call", "european", n) - bs)
        for n in steps
    ]
    for e_coarse, e_fine in zip(errs, errs[1:]):
        assert e_fine < e_coarse
    # O(1/n): error at 800 steps should be ~16x smaller than at 50 steps
    assert errs[-1] < errs[0] / 8.0


def test_crr_convergence_rate_fitted_exponent() -> None:
    """Fit the CRR discretisation-error exponent by log-log regression.

    CRR is a first-order scheme: error(n) ~ C / n, with an odd/even (and,
    away from at-the-money, node-alignment) oscillation superposed on the
    leading term. A single pair of step counts is not enough to *prove*
    the O(1/n) rate -- a two-point ratio can land anywhere depending on
    where each n happens to fall in the oscillation. Fitting a line
    through log|error| vs log(n) over a decade of geometrically spaced
    step counts averages the oscillation out and recovers the leading
    exponent, which must come out close to the theoretical -1.
    """
    S, K, T, r, sigma, q = 100.0, 105.0, 1.0, 0.04, 0.25, 0.01
    bs = bs_price(S, K, T, r, sigma, q, "call")
    steps = np.array([200 * 2**k for k in range(9)])  # 200 .. 51200
    errs = np.array(
        [
            abs(crr_price(S, K, T, r, sigma, q, "call", "european", int(n)) - bs)
            for n in steps
        ]
    )
    assert np.all(errs > 0.0), "tree exactly matches BS at some n -- fit is degenerate"
    slope, _intercept = np.polyfit(np.log(steps), np.log(errs), 1)
    assert -1.3 < slope < -0.7, (
        f"fitted CRR convergence exponent {slope:.3f}; theory predicts -1 "
        "(error ~ C/n) -- this is more than a loose oscillation, investigate"
    )


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
def test_american_put_geq_european_put(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> None:
    amer = crr_price(S, K, T, r, sigma, q, "put", "american", 400)
    euro = crr_price(S, K, T, r, sigma, q, "put", "european", 400)
    assert amer >= euro - 1e-12


def test_american_call_no_dividend_equals_european() -> None:
    """With q=0 early exercise of a call is never optimal (Merton)."""
    for r in (0.0, 0.05, 0.10):
        amer = crr_price(100, 100, 1, r, 0.25, 0.0, "call", "american", 500)
        euro = crr_price(100, 100, 1, r, 0.25, 0.0, "call", "european", 500)
        assert amer == pytest.approx(euro, abs=1e-10)


def test_american_call_with_dividend_carries_premium() -> None:
    premium = early_exercise_premium(100, 90, 2.0, 0.02, 0.2, 0.06, "call", 500)
    assert premium > 1e-3


@pytest.mark.parametrize(("S", "K", "T", "r", "sigma", "q"), CASES)
@pytest.mark.parametrize("otype", ["call", "put"])
def test_early_exercise_premium_nonnegative(
    S: float, K: float, T: float, r: float, sigma: float, q: float, otype: str
) -> None:
    assert early_exercise_premium(S, K, T, r, sigma, q, otype, 200) >= 0.0


def test_american_put_at_least_intrinsic() -> None:
    """Deep ITM American put must be worth at least immediate exercise."""
    for S in (40.0, 60.0, 80.0):
        amer = crr_price(S, 100.0, 1.0, 0.08, 0.2, 0.0, "put", "american", 400)
        assert amer >= (100.0 - S) - 1e-10


def test_deep_itm_american_put_near_intrinsic_high_rates() -> None:
    """High carry makes early exercise almost certain: value ~ intrinsic."""
    amer = crr_price(20.0, 100.0, 1.0, 0.10, 0.15, 0.0, "put", "american", 400)
    assert amer == pytest.approx(80.0, abs=1e-6)


def test_famous_american_put_benchmark() -> None:
    """S=K=100, T=1, r=5%, sigma=20%: American put ~ 6.0896 (literature)."""
    amer = crr_price(100, 100, 1.0, 0.05, 0.20, 0.0, "put", "american", 2000)
    assert amer == pytest.approx(6.0896, abs=5e-3)


def test_tree_put_call_parity_european() -> None:
    """European tree prices satisfy parity to tree accuracy."""
    S, K, T, r, sigma, q, n = 100.0, 95.0, 1.5, 0.03, 0.25, 0.02, 800
    c = crr_price(S, K, T, r, sigma, q, "call", "european", n)
    p = crr_price(S, K, T, r, sigma, q, "put", "european", n)
    assert c - p == pytest.approx(
        S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-9
    )


def test_single_step_tree_by_hand() -> None:
    """n=1 CRR reproduces the hand-computed one-step value."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    u = math.exp(sigma)
    d = 1 / u
    p = (math.exp(r) - d) / (u - d)
    expected = math.exp(-r) * p * (S * u - K)
    assert crr_price(S, K, T, r, sigma, 0.0, "call", "european", 1) == pytest.approx(
        expected, abs=1e-12
    )


def test_invalid_exercise_and_steps_raise() -> None:
    with pytest.raises(ValueError, match="exercise"):
        crr_price(100, 100, 1, 0.05, 0.2, 0.0, "call", "bermudan", 100)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_steps"):
        crr_price(100, 100, 1, 0.05, 0.2, 0.0, "call", "european", 0)
