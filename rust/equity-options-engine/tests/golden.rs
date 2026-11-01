//! Golden vectors: the engine must match the mpmath oracle (generated in
//! python/equity/01-options-pricing/tests/golden/) within the per-output
//! absolute tolerance carried by each record. Python and C++ consume the
//! same JSON.

use std::collections::HashMap;

use eq_options_engine::golden::{GoldenRecord, RECORDS};
use eq_options_engine::{
    black76_greeks, bs_greeks, bs_price, crr_price, implied_vol, Exercise,
};

fn evaluate(r: &GoldenRecord) -> HashMap<&'static str, f64> {
    let x = &r.inputs;
    let ot = r.option_type;
    let mut m = HashMap::new();
    match r.func {
        "bs" if r.outputs.len() == 1 => {
            m.insert("price", bs_price(x[0], x[1], x[2], x[3], x[5], x[4], ot).unwrap());
        }
        "bs" => {
            let g = bs_greeks(x[0], x[1], x[2], x[3], x[5], x[4], ot).unwrap();
            for (k, v) in [("price", g.price), ("delta", g.delta), ("gamma", g.gamma),
                           ("vega", g.vega), ("theta", g.theta), ("rho", g.rho),
                           ("vanna", g.vanna), ("volga", g.volga)] {
                m.insert(k, v);
            }
        }
        "iv" => {
            m.insert("sigma", implied_vol(x[0], x[1], x[2], x[3], x[4], x[5], ot).unwrap());
        }
        "crr" => {
            let ex = if r.flag == 1 { Exercise::American } else { Exercise::European };
            m.insert("price", crr_price(x[0], x[1], x[2], x[3], x[5], x[4], ot, ex, x[6] as usize).unwrap());
        }
        "b76" => {
            let g = black76_greeks(x[0], x[1], x[2], x[3], x[4], ot).unwrap();
            for (k, v) in [("price", g.price), ("delta", g.delta), ("gamma", g.gamma),
                           ("vega", g.vega), ("theta", g.theta), ("rho", g.rho)] {
                m.insert(k, v);
            }
        }
        other => panic!("unknown golden function {other}"),
    }
    m
}

#[test]
fn golden_coverage() {
    assert!(RECORDS.len() >= 400);
    for f in ["bs", "iv", "crr", "b76"] {
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
