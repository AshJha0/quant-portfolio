"""Extreme-parameter and degenerate-input behaviour (contract item 6).

Long maturities, huge/tiny strikes, the ill-conditioned corners of the
implied-vol inversion, and the smallest legal Monte Carlo samples. Every
case here is also written up in docs/VALIDATION.md.
"""
import math

import pytest

from eq_bs_replication import (
    call_greeks,
    call_price,
    implied_volatility,
    mc_call_price,
    put_greeks,
    put_price,
)


# ---------------------------------------------------------------------
# Put-side no-arbitrage bounds, systematically
# ---------------------------------------------------------------------
@pytest.mark.parametrize("S", [1.0, 50.0, 100.0, 1_000.0])
@pytest.mark.parametrize("K", [50.0, 100.0, 200.0])
@pytest.mark.parametrize("r", [-0.02, 0.0, 0.05])
@pytest.mark.parametrize("T", [0.01, 1.0, 30.0])
def test_put_bounds_hold_across_a_wide_grid(S, K, r, T):
    """``max(K e^{-rT} - S, 0) <= P <= K e^{-rT}`` for every contract.

    The lower bound is the discounted intrinsic (a European put cannot be
    worth less than exercising at expiry, discounted); the upper bound is
    the discounted strike (the most the put can ever pay is K, if the
    stock goes to zero). Checked across sign of r, three orders of
    magnitude of moneyness and maturities from 4 days to 30 years."""
    disc_k = K * math.exp(-r * T)
    for sigma in (0.05, 0.3, 1.5):
        p = put_price(S, K, r, sigma, T)
        assert p >= max(disc_k - S, 0.0) - 1e-9
        assert p <= disc_k + 1e-9


@pytest.mark.parametrize("T", [0.01, 1.0, 30.0])
def test_put_bound_is_tight_and_saturates_at_extreme_vol(T):
    """The upper bound is tight: as sigma -> infinity the put tends to the
    discounted strike (the stock is worth nothing in the limit). At
    sigma=8 the remaining gap ``S*N(-d1)`` is of order 1e-100 for long
    maturities, i.e. below double-precision resolution next to a
    ~40-point price, so the computed put is *exactly* its upper bound.

    That saturation is correct behaviour, not a bound violation -- but it
    means "put strictly below discounted strike" is not a property code
    can rely on at extreme volatilities, which is worth pinning.

    Note the limit is governed by TOTAL volatility ``sigma*sqrt(T)``, not
    ``sigma``: sigma=8 over four days is a 0.8 total vol and nowhere near
    the bound, while the same sigma over 30 years is a total vol of 44 and
    saturates completely. The test therefore fixes total vol, not sigma."""
    S, K, r = 100.0, 100.0, 0.03
    disc_k = K * math.exp(-r * T)
    sigma = 8.0 / math.sqrt(T)  # total volatility of 8.0 in every case
    p = put_price(S, K, r, sigma, T)
    assert p <= disc_k
    assert p > 0.9 * disc_k
    # Moderate total vol leaves a resolvable gap; extreme vol may not.
    assert put_price(S, K, r, 0.3 / math.sqrt(T), T) < disc_k


def test_deep_itm_put_approaches_its_lower_bound():
    """Deep ITM, low vol: the put is worth essentially its discounted
    intrinsic value and nothing more."""
    S, K, r, T = 10.0, 200.0, 0.03, 0.5
    disc_k = K * math.exp(-r * T)
    p = put_price(S, K, r, 0.05, T)
    assert p == pytest.approx(disc_k - S, rel=1e-9)


# ---------------------------------------------------------------------
# Very long maturities
# ---------------------------------------------------------------------
@pytest.mark.parametrize("T", [10.0, 30.0, 50.0])
def test_long_dated_contracts_stay_finite_and_satisfy_parity(T):
    """A 30-year option is a real instrument (LEAPS, structured notes,
    pension hedges). Nothing may overflow, and put-call parity must hold
    to the same tolerance as at one year."""
    S, K, r, sigma = 100.0, 100.0, 0.03, 0.2
    c = call_price(S, K, r, sigma, T)
    p = put_price(S, K, r, sigma, T)
    assert math.isfinite(c) and math.isfinite(p)
    assert (c - p) == pytest.approx(S - K * math.exp(-r * T), abs=1e-9)


def test_thirty_year_call_is_dominated_by_discounting_not_optionality():
    """At T=30 with r=3%, the strike discounts to about 40% of spot, so a
    struck-at-spot call is worth well over half the stock: most of its
    value is the deferred payment, not the option. Pinned because it is
    the intuition long-dated quotes constantly violate."""
    S, K, r, sigma, T = 100.0, 100.0, 0.03, 0.2, 30.0
    c = call_price(S, K, r, sigma, T)
    assert 0.6 * S < c < S
    assert K * math.exp(-r * T) == pytest.approx(40.657, abs=1e-3)


@pytest.mark.parametrize("T", [10.0, 30.0])
def test_long_dated_greeks_are_finite_and_correctly_signed(T):
    S, K, r, sigma = 100.0, 100.0, 0.03, 0.2
    g = call_greeks(S, K, r, sigma, T)
    pg = put_greeks(S, K, r, sigma, T)
    for value in (g.delta, g.gamma, g.vega, g.theta, g.rho):
        assert math.isfinite(value)
    assert 0.0 < g.delta < 1.0
    assert g.gamma > 0.0
    assert g.vega > 0.0
    assert g.rho > 0.0
    assert pg.delta == pytest.approx(g.delta - 1.0, abs=1e-12)
    # Gamma decays with maturity: a 30-year option has almost no convexity
    # per unit of spot, so delta hedging it is nearly static.
    assert g.gamma < call_greeks(S, K, r, sigma, 1.0).gamma


def test_long_dated_monte_carlo_still_matches_the_closed_form():
    S, K, r, sigma, T = 100.0, 100.0, 0.03, 0.2, 30.0
    closed = call_price(S, K, r, sigma, T)
    est, se = mc_call_price(S, K, r, sigma, T, n_paths=400_000, seed=5)
    assert abs(est - closed) < 3 * se


# ---------------------------------------------------------------------
# Huge and tiny strikes / spots: exp and log at the edges
# ---------------------------------------------------------------------
@pytest.mark.parametrize("K", [1e-10, 1e-6, 1e6, 1e10])
def test_extreme_strikes_do_not_overflow_and_respect_bounds(K):
    """``log(S/K)`` spans +/-23 for these strikes and ``d1`` runs to
    +/-100 or beyond. ``math.erf`` saturates cleanly at +/-1 there, so the
    prices must still be finite and inside their no-arbitrage bounds."""
    S, r, sigma, T = 100.0, 0.03, 0.2, 1.0
    c = call_price(S, K, r, sigma, T)
    p = put_price(S, K, r, sigma, T)
    disc_k = K * math.exp(-r * T)
    assert math.isfinite(c) and math.isfinite(p)
    assert max(S - disc_k, 0.0) - 1e-6 <= c <= S + 1e-6
    assert max(disc_k - S, 0.0) - 1e-6 <= p <= disc_k + 1e-6


@pytest.mark.parametrize("K", [1e-10, 1e10])
def test_extreme_strikes_still_satisfy_put_call_parity_in_relative_terms(K):
    """Parity is checked *relatively* here: at K=1e10 the two sides are
    ~1e10, so an absolute 1e-9 tolerance would be meaningless."""
    S, r, sigma, T = 100.0, 0.03, 0.2, 1.0
    c, p = call_price(S, K, r, sigma, T), put_price(S, K, r, sigma, T)
    rhs = S - K * math.exp(-r * T)
    assert (c - p) == pytest.approx(rhs, rel=1e-12, abs=1e-9)


def test_vanishing_strike_call_is_the_stock_itself():
    """As K -> 0 a call becomes the stock (certain exercise, no cost)."""
    assert call_price(100.0, 1e-12, 0.03, 0.2, 1.0) == pytest.approx(100.0, rel=1e-12)
    assert put_price(100.0, 1e-12, 0.03, 0.2, 1.0) == pytest.approx(0.0, abs=1e-12)


def test_enormous_strike_put_is_the_discounted_strike_minus_stock():
    K, r, T = 1e10, 0.03, 1.0
    p = put_price(100.0, K, r, 0.2, T)
    assert p == pytest.approx(K * math.exp(-r * T) - 100.0, rel=1e-12)
    assert call_price(100.0, K, r, 0.2, T) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("sigma", [1e-6, 1e-3, 5.0, 20.0])
def test_extreme_volatilities_stay_within_bounds(sigma):
    S, K, r, T = 100.0, 100.0, 0.03, 1.0
    c = call_price(S, K, r, sigma, T)
    assert max(S - K * math.exp(-r * T), 0.0) - 1e-9 <= c <= S + 1e-9


def test_discount_factor_overflow_raises_informative_value_error():
    """``exp(-r*T)`` overflows for a sufficiently negative ``r*T``. The
    bare ``OverflowError: math range error`` from ``math.exp`` named
    neither the input nor the reason; it is now a ValueError that does."""
    with pytest.raises(ValueError, match="overflows double precision"):
        call_price(100.0, 100.0, -10.0, 0.2, 100.0)
    with pytest.raises(ValueError, match="overflows double precision"):
        put_price(100.0, 100.0, -1.0, 0.2, 1_000.0)


# ---------------------------------------------------------------------
# Implied vol at the no-arbitrage boundaries (ill-conditioning)
# ---------------------------------------------------------------------
def test_implied_vol_at_exactly_intrinsic_is_ill_conditioned_not_wrong():
    """A call quoted at exactly its discounted intrinsic value implies a
    volatility of zero -- but vega there is so small that the routine
    returns a sigma of order 1e-2 which still reprices to within 1e-8.

    The number is correct to the tolerance it promises and useless as a
    volatility. This test pins both halves of that statement, because the
    failure mode on a desk is treating such a quote as a real surface
    point rather than dropping it."""
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    iv = implied_volatility(intrinsic, S, K, r, T)
    # It reprices correctly...
    assert call_price(S, K, r, iv, T) == pytest.approx(intrinsic, abs=1e-7)
    # ...while being nowhere near the true implied vol of 0.
    assert 0.0 < iv < 0.05
    # And vega at that point is negligible: the whole vol range below 5%
    # is worth less than a cent of price, which is why it is unidentified.
    assert call_greeks(S, K, r, iv, T).vega < 1.0


def test_implied_vol_just_below_intrinsic_is_rejected():
    """One tick below the bound is arbitrage, not a low volatility."""
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_volatility(intrinsic - 0.01, S, K, r, T)


def test_implied_vol_at_exactly_the_upper_bound_returns_a_large_finite_number():
    """``C = S`` implies infinite volatility. The routine cannot return
    infinity, so it returns the first large finite sigma that reprices to
    tolerance -- documented, tested, and not to be quoted."""
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    iv = implied_volatility(S, S, K, r, T)
    assert math.isfinite(iv)
    assert iv > 5.0
    assert call_price(S, K, r, iv, T) == pytest.approx(S, abs=1e-6)


def test_implied_vol_inverts_a_quote_above_five_hundred_percent():
    """The bisection bracket used to stop at sigma=5.0. A quote implying
    900% vol (a real thing in a distressed name or a crypto option) must
    be inverted, not clamped to the bracket edge."""
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    price = call_price(S, K, r, 9.0, T)
    assert implied_volatility(price, S, K, r, T) == pytest.approx(9.0, rel=1e-5)


def test_implied_vol_rejects_non_positive_time_to_expiry():
    with pytest.raises(ValueError, match="T must be strictly positive"):
        implied_volatility(5.0, 100.0, 100.0, 0.05, 0.0)


@pytest.mark.parametrize("T", [1e-4, 30.0])
def test_implied_vol_round_trips_at_extreme_maturities(T):
    S, K, r = 100.0, 100.0, 0.03
    for sigma in (0.1, 0.4, 1.0):
        price = call_price(S, K, r, sigma, T)
        assert implied_volatility(price, S, K, r, T) == pytest.approx(sigma, rel=1e-4)


# ---------------------------------------------------------------------
# Monte Carlo at the smallest legal sample sizes
# ---------------------------------------------------------------------
def test_mc_single_path_with_antithetic_is_rejected():
    """``n_paths=1`` with antithetic pairing means ``1 // 2 == 0`` draws:
    the old code returned ``(nan, nan)`` after a RuntimeWarning."""
    with pytest.raises(ValueError, match="antithetic sampling needs n_paths >= 2"):
        mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=1, seed=1)


def test_mc_single_path_without_antithetic_prices_but_has_no_error_bar():
    """One draw is a legal (if useless) estimate: a price with no
    measurable spread, so the standard error is NaN rather than 0."""
    price, se = mc_call_price(
        100.0, 100.0, 0.03, 0.2, 1.0, n_paths=1, seed=1, antithetic=False
    )
    assert math.isfinite(price)
    assert price >= 0.0
    assert math.isnan(se)


def test_mc_one_antithetic_pair_has_no_error_bar_either():
    """Two mirrored draws are ONE independent unit, so ddof=1 across pair
    means is undefined -- NaN, not a falsely tight error bar. (The old
    code reported a standard error equal to the price itself here.)"""
    price, se = mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=2, seed=1)
    assert math.isfinite(price)
    assert math.isnan(se)


def test_mc_odd_path_count_rounds_down_to_whole_pairs():
    """``n_paths=101`` simulates 100 paths (50 mirrored pairs); the result
    must be identical to asking for 100 with the same seed."""
    a = mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=101, seed=3)
    b = mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=100, seed=3)
    assert a == b


def test_mc_odd_path_count_without_antithetic_uses_every_path():
    a = mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=101, seed=3,
                      antithetic=False)
    b = mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=100, seed=3,
                      antithetic=False)
    assert a != b


def test_mc_three_paths_is_one_pair():
    a = mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=3, seed=3)
    b = mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=2, seed=3)
    assert a[0] == b[0]
    assert math.isnan(a[1]) and math.isnan(b[1])


@pytest.mark.parametrize("n_paths", [0, -10])
def test_mc_rejects_non_positive_path_counts(n_paths):
    with pytest.raises(ValueError, match="n_paths must be"):
        mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=n_paths)


def test_mc_rejects_non_integer_path_count():
    with pytest.raises(ValueError, match="n_paths must be an int"):
        mc_call_price(100.0, 100.0, 0.03, 0.2, 1.0, n_paths=100.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 0.0},
        {"sigma": -0.2},
        {"T": 0.0},
        {"T": -1.0},
        {"S": 0.0},
        {"K": -100.0},
    ],
)
def test_mc_input_validation_mirrors_the_closed_form(kwargs):
    """The simulation would happily run at sigma=0 or T=0 (deterministic
    terminal price), but the closed form refuses those inputs on purpose.
    Both must agree on what is valid, or the two implementations are no
    longer testing the same contract."""
    args = {"S": 100.0, "K": 100.0, "r": 0.03, "sigma": 0.2, "T": 1.0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        mc_call_price(n_paths=1_000, seed=1, **args)
    with pytest.raises(ValueError):
        call_price(**args)


def test_mc_antithetic_standard_error_is_honest_about_its_own_accuracy():
    """The reported standard error must match the estimator's ACTUAL
    sampling dispersion. Treating the 2m mirrored payoffs as 2m
    independent observations (the old formula) overstated it by ~33% on
    this contract, because it ignored the negative correlation that makes
    antithetic sampling work in the first place.

    Here: run 60 independent seeds, compare the empirical standard
    deviation of the 60 price estimates against the mean reported
    standard error. They should agree to well within 20%."""
    S, K, r, sigma, T = 100.0, 100.0, 0.03, 0.2, 1.0
    results = [
        mc_call_price(S, K, r, sigma, T, n_paths=20_000, seed=s)
        for s in range(60)
    ]
    prices = [p for p, _ in results]
    mean_price = sum(prices) / len(prices)
    empirical = math.sqrt(
        sum((p - mean_price) ** 2 for p in prices) / (len(prices) - 1)
    )
    reported = sum(se for _, se in results) / len(results)
    assert abs(reported - empirical) / empirical < 0.20


def test_mc_antithetic_beats_plain_at_the_same_path_count():
    """The variance reduction itself, measured on a like-for-like basis:
    same number of simulated terminal prices, correctly computed error
    bars for both."""
    S, K, r, sigma, T = 100.0, 100.0, 0.03, 0.2, 1.0
    _, se_plain = mc_call_price(S, K, r, sigma, T, n_paths=100_000, seed=2,
                                antithetic=False)
    _, se_anti = mc_call_price(S, K, r, sigma, T, n_paths=100_000, seed=2,
                               antithetic=True)
    assert se_anti < se_plain
