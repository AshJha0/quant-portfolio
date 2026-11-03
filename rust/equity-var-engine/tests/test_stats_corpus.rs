//! W20 characterisation corpus for the shared stats layer (Round 8).
//!
//! `tests/data/stats_corpus.txt` records this crate's special functions and
//! moments on a fixed grid, captured before the stats code moved to
//! `qp_core::stats`. Every row must reproduce within
//! `1e-12 * max(1, |recorded|)`. Rows the merge moved by more than that
//! carry the pre-merge bits and the mpmath value; for those the new value
//! must be closer to mpmath than the old one (a documented wrong-answer
//! fix, see MIGRATION.md).

use eq_var_engine::stats::*;
fn b(r: Result<f64, eq_var_engine::EqVarError>) -> String {
    match r {
        Ok(v) => format!("{:016x}", v.to_bits()),
        Err(_) => "ERR".into(),
    }
}
fn f(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}
/// The grid, one `name args bits` line per value.
fn grid() -> String {
    let mut o = String::new();
    let xs = [1e-3, 0.1, 0.3, 0.5, 1.0, 1.5, 2.5, 5.0, 10.0, 50.0, 170.5];
    for x in xs {
        o += &format!("ln_gamma {x} {}\n", b(ln_gamma(x)));
    }
    let ax = [
        (0.5, 0.1),
        (0.5, 2.0),
        (1.0, 0.5),
        (2.5, 1.0),
        (2.5, 7.0),
        (10.0, 3.0),
        (10.0, 20.0),
        (125.0, 130.0),
    ];
    for (a, x) in ax {
        o += &format!("gamma_p {a} {x} {}\n", b(regularized_gamma_p(a, x)));
        o += &format!("gamma_q {a} {x} {}\n", b(regularized_gamma_q(a, x)));
    }
    let zs = [
        -38.0, -20.0, -10.0, -8.0, -6.0, -3.0, -1.5, -0.3, 0.0, 0.2, 1.0, 2.33, 4.0, 7.0, 9.0,
    ];
    for z in zs {
        o += &format!("normal_cdf {z} {}\n", f(normal_cdf(z)));
        o += &format!("normal_pdf {z} {}\n", f(normal_pdf(z)));
    }
    let ps = [
        1e-12, 1e-10, 1e-6, 1e-4, 0.001, 0.0025, 0.01, 0.025, 0.05, 0.1, 0.3, 0.5, 0.7, 0.95, 0.99,
        0.999, 0.999999,
    ];
    for p in ps {
        o += &format!("normal_ppf {p} {}\n", b(normal_ppf(p)));
    }
    let abx = [
        (0.5, 0.5, 0.3),
        (2.0, 3.0, 0.4),
        (5.0, 0.5, 0.9),
        (248.0, 3.0, 0.99),
        (0.5, 2.5, 0.01),
    ];
    for (a, bb, x) in abx {
        o += &format!("beta {a} {bb} {x} {}\n", b(regularized_beta(a, bb, x)));
    }
    for (x, df) in [
        (0.0, 1.0),
        (0.5, 1.0),
        (3.84, 1.0),
        (10.0, 2.0),
        (25.0, 1.0),
        (60.0, 1.0),
        (5.0, 7.0),
    ] {
        o += &format!("chi2_sf {x} {df} {}\n", b(chi2_sf(x, df)));
    }
    for k in 0..15u64 {
        o += &format!("binom {k} 250 0.01 {}\n", b(binomial_cdf(k, 250, 0.01)));
    }
    let dfs = [2.5, 3.0, 4.0, 5.0, 6.0, 10.0, 30.0, 200.0];
    for df in dfs {
        for x in [-6.0, -2.0, -0.5, 0.0, 1.0, 3.0] {
            o += &format!("t_pdf {x} {df} {}\n", b(student_t_pdf(x, df)));
            o += &format!("t_cdf {x} {df} {}\n", b(student_t_cdf(x, df)));
        }
        for p in [1e-6, 0.001, 0.0025, 0.01, 0.025, 0.05, 0.2, 0.5, 0.9, 0.99] {
            o += &format!("t_ppf {p} {df} {}\n", b(student_t_ppf(p, df)));
        }
    }
    let s: Vec<f64> = (0..257)
        .map(|i| ((i as f64) * 0.7311).sin() * 1234.5 + ((i * i) % 17) as f64 - 3.25)
        .collect();
    o += &format!("mean {}\n", b(mean(&s)));
    o += &format!("std {}\n", b(stdev(&s)));
    o += &format!("popvar {}\n", b(population_variance(&s)));
    o += &format!("skew {}\n", b(skewness(&s)));
    o += &format!("kurt {}\n", b(excess_kurtosis(&s)));
    o
}

fn bits(h: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(h, 16).expect("hex bits"))
}

#[test]
fn stats_corpus_reproduces() {
    let now = grid();
    let data = include_str!("data/stats_corpus.txt");
    let rows: Vec<&str> = data.lines().filter(|l| !l.starts_with('#')).collect();
    let got: Vec<&str> = now.lines().collect();
    assert_eq!(rows.len(), got.len(), "grid and corpus differ in length");
    let (mut fixes, mut worst) = (0, 0.0_f64);
    for (row, g) in rows.iter().zip(&got) {
        let r: Vec<&str> = row.split_whitespace().collect();
        let g: Vec<&str> = g.split_whitespace().collect();
        let k = g.len() - 1;
        assert_eq!(r[..k], g[..k], "row keys differ: {row}");
        if r[k] == "ERR" || g[k] == "ERR" {
            assert_eq!(r[k], g[k], "error status changed: {row}");
            continue;
        }
        let (want, have) = (bits(r[k]), bits(g[k]));
        let scaled = (have - want).abs() / want.abs().max(1.0);
        worst = worst.max(scaled);
        assert!(
            scaled <= 1e-12,
            "{row}: now {have:e}, scaled diff {scaled:e}"
        );
        if let Some(i) = r.iter().position(|t| *t == "was") {
            let old = bits(r[i + 1]);
            let truth: f64 = r[i + 3].parse().expect("truth");
            assert!(
                (have - truth).abs() < (old - truth).abs(),
                "{row}: the fix must move toward the mpmath value"
            );
            fixes += 1;
        }
    }
    println!(
        "{} rows, {fixes} documented fixes, worst scaled diff {worst:e}",
        rows.len()
    );
}
