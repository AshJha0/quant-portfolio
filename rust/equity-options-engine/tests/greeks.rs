//! Analytic vs finite-difference Greeks, and Greek identities.

mod common;

use eq_options_engine::{bs_greeks, bs_price, fd_greeks, OptionType};

#[test]
fn analytic_matches_finite_difference_to_1e6() {
    // Relative tolerance 1e-6, as in the C++ engine's gate.
    for &(s, k, t, r, q, sigma) in &[
        (100.0, 100.0, 1.0, 0.05, 0.0, 0.2),
        (100.0, 120.0, 0.5, 0.03, 0.01, 0.3),
        (100.0, 80.0, 2.0, 0.02, 0.03, 0.15),
        (50.0, 55.0, 0.25, -0.01, 0.0, 0.4),
        (1000.0, 900.0, 1.5, 0.045, 0.015, 0.28),
    ] {
        for ot in [OptionType::Call, OptionType::Put] {
            let an = bs_greeks(s, k, t, r, sigma, q, ot).unwrap();
            let fd = fd_greeks(bs_price, s, k, t, r, sigma, q, ot).unwrap();
            // First-order Greeks (and price) to 1e-6 rel; second-order
            // Greeks use the larger h ~ eps^0.25 bump whose O(h^2)
            // truncation term justifies 5e-6.
            for (name, a, f, tol) in [
                ("price", an.price, fd.price, 1e-6),
                ("delta", an.delta, fd.delta, 1e-6),
                ("vega", an.vega, fd.vega, 1e-6),
                ("theta", an.theta, fd.theta, 1e-6),
                ("rho", an.rho, fd.rho, 1e-6),
                ("gamma", an.gamma, fd.gamma, 5e-6),
                ("vanna", an.vanna, fd.vanna, 5e-6),
                ("volga", an.volga, fd.volga, 5e-6),
            ] {
                let scale = a.abs().max(f.abs()).max(1.0);
                assert!(
                    (a - f).abs() <= tol * scale,
                    "{name} mismatch at (S={s}, K={k}, T={t}, {ot:?}): \
                     analytic {a} vs FD {f} (rel {:.3e})",
                    (a - f).abs() / scale
                );
            }
        }
    }
}

#[test]
fn fd_greeks_work_for_any_pricer_closure() {
    // Use the CRR tree as the pricer under the generic FD engine; its
    // delta should sit close to the analytic BS delta.
    use eq_options_engine::{crr_price, Exercise};
    let pricer = |s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, ot: OptionType| {
        crr_price(s, k, t, r, sigma, q, ot, Exercise::European, 800)
    };
    let fd = fd_greeks(pricer, 100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
    let an = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).unwrap();
    assert_close!(fd.delta, an.delta, 5e-3);
    assert_close!(fd.price, an.price, 5e-3);
}

#[test]
fn call_put_greek_identities() {
    let (s, k, t, r, q, sigma) = (100.0, 105.0, 0.75, 0.04, 0.02, 0.25);
    let c = bs_greeks(s, k, t, r, sigma, q, OptionType::Call).unwrap();
    let p = bs_greeks(s, k, t, r, sigma, q, OptionType::Put).unwrap();
    // Gamma, vega, vanna, volga identical for call and put:
    assert_close!(c.gamma, p.gamma, 1e-14);
    assert_close!(c.vega, p.vega, 1e-12);
    assert_close!(c.vanna, p.vanna, 1e-13);
    assert_close!(c.volga, p.volga, 1e-12);
    // delta_call - delta_put = e^{-qT} (parity in delta):
    assert_close!(c.delta - p.delta, (-q * t).exp(), 1e-13);
    // rho_call - rho_put = K T e^{-rT}:
    assert_close!(c.rho - p.rho, k * t * (-r * t).exp(), 1e-12);
}

#[test]
fn atm_call_delta_is_near_half_and_signs_are_right() {
    let g = bs_greeks(100.0, 100.0, 0.5, 0.02, 0.2, 0.0, OptionType::Call).unwrap();
    assert!(g.delta > 0.5 && g.delta < 0.65);
    assert!(g.gamma > 0.0);
    assert!(g.vega > 0.0);
    assert!(g.theta < 0.0);
    assert!(g.rho > 0.0);
    let p = bs_greeks(100.0, 100.0, 0.5, 0.02, 0.2, 0.0, OptionType::Put).unwrap();
    assert!(p.delta < 0.0 && p.delta > -0.5);
    assert!(p.rho < 0.0);
}

#[test]
fn greeks_reject_boundary_inputs() {
    assert!(bs_greeks(100.0, 100.0, 0.0, 0.05, 0.2, 0.0, OptionType::Call).is_err());
    assert!(bs_greeks(100.0, 100.0, 1.0, 0.05, 0.0, 0.0, OptionType::Call).is_err());
    assert!(bs_greeks(0.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call).is_err());
    // FD theta bump needs T > h: tiny T is rejected cleanly.
    assert!(fd_greeks(bs_price, 100.0, 100.0, 1e-6, 0.05, 0.2, 0.0, OptionType::Call).is_err());
}

#[test]
fn greeks_reject_non_finite_inputs() {
    use eq_options_engine::PricingError;
    let nan = f64::NAN;
    let inf = f64::INFINITY;
    // Every non-finite argument, at both public Greek entry points, must
    // come back as InvalidInput rather than as a struct full of NaNs. A
    // NaN Greek trips no risk limit and colours no traffic light, so it
    // is strictly worse than a loud failure.
    for (s, k, t, r, sigma, q, label) in [
        (nan, 100.0, 1.0, 0.05, 0.2, 0.0, "S=NaN"),
        (100.0, nan, 1.0, 0.05, 0.2, 0.0, "K=NaN"),
        (100.0, 100.0, nan, 0.05, 0.2, 0.0, "T=NaN"),
        (100.0, 100.0, 1.0, nan, 0.2, 0.0, "r=NaN"),
        (100.0, 100.0, 1.0, 0.05, nan, 0.0, "sigma=NaN"),
        (100.0, 100.0, 1.0, 0.05, 0.2, nan, "q=NaN"),
        (inf, 100.0, 1.0, 0.05, 0.2, 0.0, "S=+inf"),
        (100.0, 100.0, inf, 0.05, 0.2, 0.0, "T=+inf"),
        (100.0, 100.0, 1.0, -inf, 0.2, 0.0, "r=-inf"),
        (100.0, 100.0, 1.0, 0.05, inf, 0.0, "sigma=+inf"),
        (100.0, 100.0, 1.0, 0.05, 0.2, inf, "q=+inf"),
    ] {
        for ot in [OptionType::Call, OptionType::Put] {
            let an = bs_greeks(s, k, t, r, sigma, q, ot);
            assert!(
                matches!(an, Err(PricingError::InvalidInput(_))),
                "bs_greeks({label}, {ot:?}) should be InvalidInput, got {an:?}"
            );
            let fd = fd_greeks(bs_price, s, k, t, r, sigma, q, ot);
            assert!(
                matches!(fd, Err(PricingError::InvalidInput(_))),
                "fd_greeks({label}, {ot:?}) should be InvalidInput, got {fd:?}"
            );
        }
    }
    // Negative rates remain legal at both entry points.
    assert!(bs_greeks(100.0, 100.0, 1.0, -0.01, 0.2, -0.02, OptionType::Call).is_ok());
    assert!(fd_greeks(bs_price, 100.0, 100.0, 1.0, -0.01, 0.2, -0.02, OptionType::Call).is_ok());
}

#[test]
fn fd_greeks_reject_a_vol_smaller_than_the_bump() {
    use eq_options_engine::PricingError;
    // The FD bumps are floored at rel_bump * 1, so for sigma < 2e-4 the
    // down-leg of the vega/volga stencil sits at a NEGATIVE volatility.
    // A pricer that validates (bs_price) would error out; a pricer that
    // does not would return a meaningless number. Both must be rejected
    // up front, by fd_greeks itself.
    let mut called = false;
    let unvalidating = |s: f64, k: f64, t: f64, r: f64, sigma: f64, q: f64, ot: OptionType| {
        // Deliberately no validation: prices at whatever sigma it is given.
        let sqrt_t = t.sqrt();
        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;
        let n = |x: f64| 0.5 * (1.0 + libm_erf_stub(x / std::f64::consts::SQRT_2));
        Ok(match ot {
            OptionType::Call => s * (-q * t).exp() * n(d1) - k * (-r * t).exp() * n(d2),
            OptionType::Put => k * (-r * t).exp() * n(-d2) - s * (-q * t).exp() * n(-d1),
        })
    };
    for sigma in [1e-6, 1e-5, 1e-4, 2e-4] {
        let via_bs = fd_greeks(bs_price, 100.0, 100.0, 1.0, 0.05, sigma, 0.0, OptionType::Call);
        assert!(
            matches!(via_bs, Err(PricingError::InvalidInput(_))),
            "sigma={sigma}: expected InvalidInput, got {via_bs:?}"
        );
        let via_closure =
            fd_greeks(&unvalidating, 100.0, 100.0, 1.0, 0.05, sigma, 0.0, OptionType::Call);
        assert!(
            matches!(via_closure, Err(PricingError::InvalidInput(_))),
            "sigma={sigma} with a non-validating pricer: expected InvalidInput, \
             got {via_closure:?}"
        );
        called = true;
    }
    assert!(called);
    // Just above the bump the stencil is legal again and agrees with the
    // analytic vega.
    let sigma = 1e-2;
    let fd = fd_greeks(bs_price, 100.0, 100.0, 1.0, 0.05, sigma, 0.0, OptionType::Call).unwrap();
    let an = bs_greeks(100.0, 100.0, 1.0, 0.05, sigma, 0.0, OptionType::Call).unwrap();
    assert!(fd.vega.is_finite() && an.vega.is_finite());
    assert_close!(fd.vega, an.vega, 1e-4 * an.vega.abs().max(1.0));
    // Symmetric guard on spot: a spot below the delta/gamma bump is
    // rejected rather than differenced across S = 0.
    let tiny_s = fd_greeks(bs_price, 1e-5, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Call);
    assert!(matches!(tiny_s, Err(PricingError::InvalidInput(_))), "got {tiny_s:?}");
}

/// Minimal erf, only accurate enough for the non-validating test pricer above.
fn libm_erf_stub(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 — 1e-7 absolute, plenty for a test stub.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

#[test]
fn extreme_moneyness_greeks_stay_finite_and_bounded() {
    // 100x and 1/100x moneyness: delta saturates at the discounted
    // bounds, gamma/vega collapse to (positive) zero, nothing is NaN.
    let (t, r, q, sigma): (f64, f64, f64, f64) = (1.0, 0.03, 0.01, 0.2);
    let df_q = (-q * t).exp();
    for (s, k) in [(10_000.0, 100.0), (100.0, 10_000.0)] {
        for ot in [OptionType::Call, OptionType::Put] {
            let g = bs_greeks(s, k, t, r, sigma, q, ot).unwrap();
            for (name, v) in [
                ("price", g.price),
                ("delta", g.delta),
                ("gamma", g.gamma),
                ("vega", g.vega),
                ("theta", g.theta),
                ("rho", g.rho),
                ("vanna", g.vanna),
                ("volga", g.volga),
            ] {
                assert!(v.is_finite(), "{name} not finite at S={s}, K={k}, {ot:?}: {v}");
            }
            assert!(g.gamma >= 0.0 && g.vega >= 0.0);
            match ot {
                OptionType::Call => assert!(g.delta >= 0.0 && g.delta <= df_q + 1e-15),
                OptionType::Put => assert!(g.delta <= 0.0 && g.delta >= -df_q - 1e-15),
            }
        }
    }
    // Deep ITM call delta -> e^{-qT}; deep OTM call delta -> 0.
    let itm = bs_greeks(10_000.0, 100.0, t, r, sigma, q, OptionType::Call).unwrap();
    assert_close!(itm.delta, df_q, 1e-12);
    let otm = bs_greeks(100.0, 10_000.0, t, r, sigma, q, OptionType::Call).unwrap();
    assert!(otm.delta < 1e-30 && otm.delta >= 0.0, "deep OTM delta {}", otm.delta);
}
