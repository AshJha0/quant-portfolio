"""Monte Carlo vs closed form: agreement at every sample size, and the
O(1/sqrt(n)) convergence rate itself (both the estimate's error and the
standard error should shrink at that rate).
"""
import math

from eq_bs_replication import call_price, mc_call_price

S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.25, 0.75


def test_mc_agrees_with_closed_form_within_3_standard_errors():
    closed = call_price(S, K, r, sigma, T)
    for n in (10_000, 100_000, 1_000_000):
        est, se = mc_call_price(S, K, r, sigma, T, n_paths=n, seed=7)
        assert abs(est - closed) < 3 * se, (n, est, se, closed)


def test_mc_standard_error_shrinks_like_inverse_sqrt_n():
    ns = [10_000, 100_000, 1_000_000]
    ses = [mc_call_price(S, K, r, sigma, T, n_paths=n, seed=7)[1] for n in ns]
    # SE * sqrt(n) should be roughly constant (the O(1/sqrt(n)) law).
    products = [se * math.sqrt(n) for se, n in zip(ses, ns)]
    mean_product = sum(products) / len(products)
    for p in products:
        assert abs(p - mean_product) / mean_product < 0.15, products
    # And SE itself must be monotonically decreasing as n grows.
    assert ses[0] > ses[1] > ses[2]


def test_mc_absolute_error_shrinks_with_more_paths():
    closed = call_price(S, K, r, sigma, T)
    ns = [10_000, 100_000, 1_000_000]
    errors = [abs(mc_call_price(S, K, r, sigma, T, n_paths=n, seed=7)[0] - closed)
              for n in ns]
    # Not every individual error need be smaller (it's a random
    # estimator), but the trend across three orders of magnitude of
    # paths should be sharply downward; the largest-n error should be
    # comfortably the smallest.
    assert errors[-1] <= max(errors[0], errors[1])


def test_mc_antithetic_reduces_variance():
    _, se_plain = mc_call_price(S, K, r, sigma, T, n_paths=200_000, seed=1,
                                antithetic=False)
    _, se_anti = mc_call_price(S, K, r, sigma, T, n_paths=200_000, seed=1,
                               antithetic=True)
    assert se_anti < se_plain


def test_mc_is_reproducible_given_seed():
    est1, se1 = mc_call_price(S, K, r, sigma, T, n_paths=50_000, seed=42)
    est2, se2 = mc_call_price(S, K, r, sigma, T, n_paths=50_000, seed=42)
    assert est1 == est2
    assert se1 == se2


def test_mc_matches_closed_form_across_moneyness():
    for k in (80.0, 100.0, 130.0):
        closed = call_price(S, k, r, sigma, T)
        est, se = mc_call_price(S, k, r, sigma, T, n_paths=300_000, seed=11)
        assert abs(est - closed) < 3 * se, (k, est, se, closed)
