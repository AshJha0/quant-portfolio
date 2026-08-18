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
