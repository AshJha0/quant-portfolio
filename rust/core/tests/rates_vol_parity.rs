//! Parity of the Rust rates-vol port with the Python reference values in
//! `data/rates_vol_parity.txt` (written by
//! `python/equity/04-fixed-income/tools/gen_rates_vol_parity.py`).

use qp_core::rates_vol::{Variant, bermudan_tree, sabr_vol};

fn floats(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect()
}

#[test]
fn matches_python() {
    let text = include_str!("data/rates_vol_parity.txt");
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .collect();
    let (mut n_s, mut n_b, mut i) = (0, 0, 0);
    while i < lines.len() {
        let p: Vec<&str> = lines[i].split_whitespace().collect();
        if p[0] == "S" {
            let x: Vec<f64> = p[1..9].iter().map(|s| s.parse().unwrap()).collect();
            let variant = if p[9] == "hagan" {
                Variant::Hagan
            } else {
                Variant::Obloj
            };
            let want: f64 = p[10].parse().unwrap();
            let got = sabr_vol(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], variant).unwrap();
            // 1e-12 relative, the repository's cross-language parity bound:
            // the Obloj branch calls ln(), whose last bits differ between the
            // libm that wrote the fixture and the runner's (CI saw 6.6 ulp).
            assert!(
                (got - want).abs() <= 1e-12 * want.abs(),
                "{}: {got} vs {want}",
                lines[i]
            );
            n_s += 1;
            i += 1;
        } else {
            assert_eq!(p[0], "B");
            let (a, sigma): (f64, f64) = (p[1].parse().unwrap(), p[2].parse().unwrap());
            let payer = p[3] == "1";
            let want: f64 = p[4].parse().unwrap();
            let t = floats(lines[i + 1]);
            let d = floats(lines[i + 2]);
            let c = floats(lines[i + 3]);
            let e: Vec<bool> = lines[i + 4]
                .split_whitespace()
                .skip(1)
                .map(|s| s == "1")
                .collect();
            let got = bermudan_tree(a, sigma, &t, &d, &c, &e, payer).unwrap();
            assert!((got - want).abs() <= 1e-13, "bermudan {got} vs {want}");
            n_b += 1;
            i += 5;
        }
    }
    assert!(n_s >= 80 && n_b == 3, "{n_s} sabr rows, {n_b} bermudans");
}
