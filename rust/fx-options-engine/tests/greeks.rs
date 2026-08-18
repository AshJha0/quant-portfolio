//! Analytic Greeks vs finite differences, rho signs, vanna/volga.

mod common;
use common::assert_close;

use fx_options_engine::greeks::{central_difference, gamma, second_central_difference, vanna, vega, volga};
use fx_options_engine::{
    analytic_greeks, finite_difference_greeks, gk_price, OptionType,
};

const MKT: (f64, f64, f64, f64, f64, f64) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);

#[test]
fn finite_differences_match_analytic_to_1e6() {
    let (s, k, t, rd, rf, sig) = MKT;
    for ty in [OptionType::Call, OptionType::Put] {
        let a = analytic_greeks(s, k, t, rd, rf, sig, ty).unwrap();
        let fd = finite_difference_greeks(s, k, t, rd, rf, sig, ty, 1e-5).unwrap();
        assert_close(fd.delta_spot, a.delta_spot, 1e-6, "FD delta");
        assert_close(fd.gamma, a.gamma, 1e-4, "FD gamma"); // 2nd order stencil
        assert_close(fd.vega, a.vega, 1e-6, "FD vega");
        assert_close(fd.theta, a.theta, 1e-6, "FD theta");
        assert_close(fd.rho_domestic, a.rho_domestic, 1e-6, "FD rho_d");
        assert_close(fd.rho_foreign, a.rho_foreign, 1e-6, "FD rho_f");
        assert_close(fd.vanna, a.vanna, 1e-5, "FD vanna");
        assert_close(fd.volga, a.volga, 1e-4, "FD volga");
    }
}

#[test]
fn rho_signs_two_rate_structure() {
    let (s, k, t, rd, rf, sig) = MKT;
    let call = analytic_greeks(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
    let put = analytic_greeks(s, k, t, rd, rf, sig, OptionType::Put).unwrap();
    // Call: higher domestic rate lifts the forward -> rho_d > 0; higher
    // foreign rate is a larger dividend on the base -> rho_f < 0.
    assert!(call.rho_domestic > 0.0 && call.rho_foreign < 0.0);
    // Put: the mirror image.
    assert!(put.rho_domestic < 0.0 && put.rho_foreign > 0.0);
}

#[test]
fn vega_and_gamma_positive_and_type_independent() {
    let (s, k, t, rd, rf, sig) = MKT;
    let call = analytic_greeks(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
    let put = analytic_greeks(s, k, t, rd, rf, sig, OptionType::Put).unwrap();
    assert!(call.vega > 0.0 && call.gamma > 0.0);
    assert_close(call.vega, put.vega, 1e-15, "vega call == put");
    assert_close(call.gamma, put.gamma, 1e-15, "gamma call == put");
    assert_close(call.vanna, put.vanna, 1e-15, "vanna call == put");
    assert_close(call.volga, put.volga, 1e-15, "volga call == put");
}

#[test]
fn standalone_greeks_match_struct_fields() {
    let (s, k, t, rd, rf, sig) = MKT;
    let a = analytic_greeks(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
    assert_close(gamma(s, k, t, rd, rf, sig).unwrap(), a.gamma, 1e-15, "gamma");
    assert_close(vega(s, k, t, rd, rf, sig).unwrap(), a.vega, 1e-15, "vega");
    assert_close(vanna(s, k, t, rd, rf, sig).unwrap(), a.vanna, 1e-15, "vanna");
    assert_close(volga(s, k, t, rd, rf, sig).unwrap(), a.volga, 1e-15, "volga");
}

#[test]
fn generic_fd_helpers_reproduce_delta_and_gamma() {
    let (s, k, t, rd, rf, sig) = MKT;
    let a = analytic_greeks(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
    let price_at_spot = |x: f64| gk_price(x, k, t, rd, rf, sig, OptionType::Call);
    let d = central_difference(price_at_spot, s, s * 1e-5).unwrap();
    let g = second_central_difference(price_at_spot, s, s * 1e-4).unwrap();
    assert_close(d, a.delta_spot, 1e-6, "generic FD delta");
    assert_close(g, a.gamma, 1e-4, "generic FD gamma");
}

#[test]
fn otm_option_theta_negative() {
    let (s, k, t, rd, rf, sig) = MKT;
    let g = analytic_greeks(s, k, t, rd, rf, sig, OptionType::Call).unwrap();
    assert!(g.theta < 0.0, "OTM call theta {} should be negative", g.theta);
}

#[test]
fn greeks_require_positive_t_and_sigma() {
    let (s, k, _, rd, rf, sig) = MKT;
    assert!(analytic_greeks(s, k, 0.0, rd, rf, sig, OptionType::Call).is_err());
    assert!(analytic_greeks(s, k, 0.5, rd, rf, 0.0, OptionType::Call).is_err());
    assert!(gamma(s, k, 0.0, rd, rf, sig).is_err());
}
