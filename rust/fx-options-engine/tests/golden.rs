//! Golden vectors: the engine must match the mpmath oracle (generated in
//! python/fx/01-options-pricing/tests/golden/) within the per-output
//! absolute tolerance carried by each record. Python and C++ consume the
//! same JSON.

use std::collections::HashMap;

use fx_options_engine::golden::{GoldenRecord, RECORDS};
use fx_options_engine::{
    analytic_greeks, atm_dns_strike, binomial_price, cip_forward, delta, gk_price, implied_vol,
    strike_from_delta, DeltaConvention, Exercise,
};

const CONVENTIONS: [DeltaConvention; 4] = [
    DeltaConvention::Spot,
    DeltaConvention::Forward,
    DeltaConvention::SpotPa,
    DeltaConvention::ForwardPa,
];

fn evaluate(r: &GoldenRecord) -> HashMap<&'static str, f64> {
    let x = &r.inputs;
    let ot = r.option_type;
    let mut m = HashMap::new();
    match r.func {
        "gk" if r.outputs.len() == 1 => {
            m.insert("price", gk_price(x[0], x[1], x[2], x[3], x[4], x[5], ot).unwrap());
        }
        "gk" => {
            let g = analytic_greeks(x[0], x[1], x[2], x[3], x[4], x[5], ot).unwrap();
            let pa = |c| delta(x[0], x[1], x[2], x[3], x[4], x[5], ot, c).unwrap();
            for (k, v) in [
                ("price", g.price),
                ("delta_spot", g.delta_spot),
                ("delta_fwd", g.delta_forward),
                ("delta_spot_pa", pa(DeltaConvention::SpotPa)),
                ("delta_fwd_pa", pa(DeltaConvention::ForwardPa)),
                ("gamma", g.gamma),
                ("vega", g.vega),
                ("theta", g.theta),
                ("rho_d", g.rho_domestic),
                ("rho_f", g.rho_foreign),
                ("vanna", g.vanna),
                ("volga", g.volga),
            ] {
                m.insert(k, v);
            }
        }
        "iv" => {
            m.insert("sigma", implied_vol(x[0], x[1], x[2], x[3], x[4], x[5], ot).unwrap());
        }
        "strike" => {
            let conv = CONVENTIONS[r.flag as usize];
            m.insert("K", strike_from_delta(x[0], x[1], x[2], x[3], x[4], x[5], ot, conv).unwrap());
        }
        "dns_atm" => {
            let conv = if r.flag == 1 { DeltaConvention::SpotPa } else { DeltaConvention::Spot };
            m.insert("K", atm_dns_strike(x[0], x[1], x[2], x[3], x[4], conv).unwrap());
        }
        "crr" => {
            let ex = if r.flag == 1 { Exercise::American } else { Exercise::European };
            m.insert("price", binomial_price(x[0], x[1], x[2], x[3], x[4], x[5], ot, x[6] as u32, ex).unwrap());
        }
        "fwd" => {
            m.insert("forward", cip_forward(x[0], x[1], x[2], x[3]).unwrap());
        }
        other => panic!("unknown golden function {other}"),
    }
    m
}

#[test]
fn golden_coverage() {
    assert!(RECORDS.len() >= 400);
    for f in ["gk", "iv", "strike", "dns_atm", "crr", "fwd"] {
        assert!(RECORDS.iter().any(|r| r.func == f), "{f}");
    }
}

#[test]
fn all_records_match_oracle_and_track_worst_deviation() {
    let mut worst = 0.0_f64;
    for (i, r) in RECORDS.iter().enumerate() {
        let got = evaluate(r);
        for &(name, value, tol) in r.outputs {
            let v = got[name];
            assert!(
                (v - value).abs() <= tol,
                "record {i} ({}) {name}: {v} vs {value} (tol {tol:e})",
                r.func
            );
            worst = worst.max((v - value).abs() / tol);
        }
    }
    println!("worst deviation / tolerance = {worst:.3}");
    assert!(worst <= 1.0);
}
