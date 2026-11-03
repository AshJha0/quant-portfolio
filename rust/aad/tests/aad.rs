//! W19 in Rust: adjoint Greeks against closed-form Greeks and central bumps,
//! the same oracles as `python/core/tests/test_aad_python.py`.
#![allow(clippy::needless_range_loop, clippy::collapsible_if)]

use qp_aad::mc::{self, Payoff};
use qp_aad::swap::{self, Swap};
use qp_aad::{Real, Tape, bs};
use qp_rates_curves::{FixedPeriod, FloatPeriod, Quote, bootstrap_dated};
use std::path::PathBuf;

const BASE: [f64; 6] = [100.0, 100.0, 1.0, 0.05, 0.02, 0.20];

fn close(a: f64, b: f64, rel: f64, floor: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + floor
}

fn central(f: &dyn Fn(&[f64]) -> f64, x: &[f64], i: usize, h: f64) -> f64 {
    let (mut up, mut dn) = (x.to_vec(), x.to_vec());
    up[i] += h;
    dn[i] -= h;
    (f(&up) - f(&dn)) / (2.0 * h)
}

fn ncdf(x: f64) -> f64 {
    qp_core::special::norm_cdf(x)
}

#[test]
fn tape_elementary_partials() {
    let t = Tape::new();
    let (a, b) = (t.var(1.3), t.var(0.7));
    let y = (a * b - a / b + b.rdiv(2.0) + b.rsub(3.0) - (-a)) * a.exp()
        + b.ln()
        + a.sqrt()
        + (a - b).ncdf();
    let g = t.gradient(y, &[a, b]);
    let f = |v: &[f64]| {
        let (x0, x1) = (v[0], v[1]);
        (x0 * x1 - x0 / x1 + 2.0 / x1 + 3.0 - x1 + x0) * x0.exp()
            + x1.ln()
            + x0.sqrt()
            + ncdf(x0 - x1)
    };
    assert!(close(y.value(), f(&[1.3, 0.7]), 1e-15, 0.0));
    for i in 0..2 {
        assert!(
            close(g[i], central(&f, &[1.3, 0.7], i, 1e-6), 1e-8, 0.0),
            "{i}"
        );
    }
}

#[test]
fn rewind_keeps_input_adjoints() {
    let t = Tape::new();
    let x = t.var(2.0);
    let m = t.mark();
    t.zero_adjoints();
    for _ in 0..3 {
        t.backward(x * x, 1.0, m);
        t.rewind(m);
    }
    assert!((t.adjoint(x) - 12.0).abs() < 1e-15 && t.len() == m);
}

#[test]
#[should_panic(expected = "different tape")]
fn foreign_output_panics() {
    let (t1, t2) = (Tape::new(), Tape::new());
    let x = t2.var(1.0);
    t1.backward(x * 2.0, 1.0, 0);
}

/// Closed-form Greeks (Hull 2022, ch. 19), written out independently.
fn analytic(s: f64, k: f64, t: f64, r: f64, q: f64, sig: f64, call: bool) -> [f64; 6] {
    let sd = sig * t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sig * sig) * t) / sd;
    let d2 = d1 - sd;
    let (dq, dr) = ((-q * t).exp(), (-r * t).exp());
    let pdf = (-0.5 * d1 * d1).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let vega = s * dq * pdf * t.sqrt();
    let c = -s * dq * pdf * sig / (2.0 * t.sqrt());
    if call {
        let (n1, n2) = (ncdf(d1), ncdf(d2));
        [
            s * dq * n1 - k * dr * n2,
            dq * n1,
            vega,
            k * t * dr * n2,
            -s * t * dq * n1,
            c + q * s * dq * n1 - r * k * dr * n2,
        ]
    } else {
        let (m1, m2) = (ncdf(-d1), ncdf(-d2));
        [
            k * dr * m2 - s * dq * m1,
            -dq * m1,
            vega,
            -k * t * dr * m2,
            s * t * dq * m1,
            c - q * s * dq * m1 + r * k * dr * m2,
        ]
    }
}

#[test]
fn bs_matches_analytic_and_bumps() {
    for &call in &[true, false] {
        for &(s, k, t, sig) in &[
            (100.0, 100.0, 1.0, 0.2),
            (80.0, 100.0, 0.25, 0.35),
            (130.0, 100.0, 3.0, 0.15),
        ] {
            let a = bs::greeks(s, k, t, 0.05, 0.02, sig, call);
            let c = analytic(s, k, t, 0.05, 0.02, sig, call);
            let got = [a.price, a.delta, a.vega, a.rho, a.rho_q, -a.d_t];
            for i in 0..6 {
                assert!(
                    close(got[i], c[i], 1e-12, 1e-14),
                    "call={call} {i}: {} vs {}",
                    got[i],
                    c[i]
                );
            }
        }
    }
    let a = bs::greeks(BASE[0], BASE[1], BASE[2], BASE[3], BASE[4], BASE[5], true);
    let f = |v: &[f64]| bs::price(v[0], v[1], v[2], v[3], v[4], v[5], 1.0);
    let got = [a.delta, a.d_k, a.d_t, a.rho, a.rho_q, a.vega];
    for i in 0..6 {
        assert!(
            close(
                got[i],
                central(&f, &BASE, i, 1e-5 * BASE[i].abs().max(1.0)),
                1e-6,
                1e-9
            ),
            "{i}"
        );
    }
    assert!(
        (a.price - 9.227005508154036).abs() < 1e-12,
        "same price as the Python worked example"
    );
}

#[test]
fn mc_pathwise_matches_bumps_same_normals() {
    let steps = 12;
    let z = mc::normals(2000, steps, 20261103);
    for payoff in [Payoff::European, Payoff::Asian] {
        let a = mc::greeks(
            BASE[0], BASE[1], BASE[2], BASE[3], BASE[4], BASE[5], &z, steps, payoff,
        );
        let f = |v: &[f64]| mc::price(v[0], v[1], v[2], v[3], v[4], v[5], &z, steps, payoff);
        assert!(close(a.price, f(&BASE), 1e-12, 0.0));
        let got = [a.delta, a.d_k, a.d_t, a.rho, a.rho_q, a.vega];
        for i in 0..6 {
            let b = central(&f, &BASE, i, 1e-6 * BASE[i].abs().max(1.0));
            assert!(
                close(got[i], b, 1e-6, 1e-9),
                "{payoff:?} {i}: {} vs {b}",
                got[i]
            );
        }
    }
}

#[test]
fn mc_european_converges_to_black_scholes() {
    let z = mc::normals(20000, 1, 7);
    let a = mc::greeks(
        BASE[0],
        BASE[1],
        BASE[2],
        BASE[3],
        BASE[4],
        BASE[5],
        &z,
        1,
        Payoff::European,
    );
    let b = bs::greeks(BASE[0], BASE[1], BASE[2], BASE[3], BASE[4], BASE[5], true);
    assert!((a.price - b.price).abs() < 3.0 * a.std_error);
    assert!((a.delta - b.delta).abs() < 0.01 && ((a.vega - b.vega) / b.vega).abs() < 0.03);
}

fn usd_ois() -> (Vec<Quote>, Vec<i64>, Vec<f64>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt");
    let text = std::fs::read_to_string(path).expect("fixture readable");
    let rows: Vec<Vec<&str>> = text
        .lines()
        .map(|l| {
            l.split('#')
                .next()
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>()
        })
        .filter(|r| !r.is_empty())
        .collect();
    let (mut quotes, mut days, mut dfs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut i, mut on) = (0, false);
    while i < rows.len() {
        let r = &rows[i];
        i += 1;
        match r[0] {
            "build" => on = r[1] == "usd_ois",
            _ if !on => {}
            "end" => break,
            "expect" => {
                days.push(r[1].parse().unwrap());
                dfs.push(r[2].parse().unwrap());
            }
            "fwd" => quotes.push(Quote::Forward {
                start: r[1].parse().unwrap(),
                end: r[2].parse().unwrap(),
                accrual: r[3].parse().unwrap(),
                rate: r[4].parse().unwrap(),
            }),
            "swap" => {
                let (nf, nl): (usize, usize) = (r[2].parse().unwrap(), r[3].parse().unwrap());
                let fixed = rows[i..i + nf]
                    .iter()
                    .map(|a| FixedPeriod {
                        pay: a[1].parse().unwrap(),
                        accrual: a[2].parse().unwrap(),
                    })
                    .collect();
                let floating = rows[i + nf..i + nf + nl]
                    .iter()
                    .map(|a| FloatPeriod {
                        start: a[1].parse().unwrap(),
                        end: a[2].parse().unwrap(),
                        pay: a[3].parse().unwrap(),
                        accrual: a[4].parse().unwrap(),
                    })
                    .collect();
                i += nf + nl;
                quotes.push(Quote::Swap {
                    rate: r[1].parse().unwrap(),
                    fixed,
                    floating,
                });
            }
            other => panic!("unexpected line {other}"),
        }
    }
    (quotes, days, dfs)
}

/// The periods of the SOFR OIS 7Y quote, payer, 4 % fixed, 1m notional.
fn test_swap(quotes: &[Quote]) -> Swap {
    for q in quotes {
        if let Quote::Swap {
            fixed, floating, ..
        } = q
        {
            if fixed.len() == 7 {
                return Swap {
                    fixed: fixed.clone(),
                    floating: floating.clone(),
                    rate: 0.04,
                    notional: 1e6,
                    payer: true,
                };
            }
        }
    }
    panic!("no 7Y quote")
}

#[test]
fn swap_pillar_greeks_match_bumps() {
    let (quotes, days, dfs) = usd_ois();
    assert_eq!((quotes.len(), days.len()), (18, 18));
    let sw = test_swap(&quotes);
    let (v, g) = swap::pillar_greeks(&days, &dfs, &sw);
    assert!(close(v, swap::pv_f64(&days, &dfs, &sw), 1e-15, 0.0));
    let x: Vec<f64> = dfs.iter().map(|d| d.ln()).collect();
    let f = |u: &[f64]| swap::pv(&days, u, &sw);
    for j in 0..x.len() {
        assert!(close(g[j], central(&f, &x, j, 1e-6), 1e-6, 1e-6), "{j}");
    }
    let rec = Swap {
        payer: false,
        ..sw.clone()
    };
    assert_eq!(swap::pv_f64(&days, &dfs, &rec), -v);
}

#[test]
fn swap_quote_greeks_match_rebootstrap() {
    let (quotes, days, _) = usd_ois();
    let sw = test_swap(&quotes);
    let base = bootstrap_dated(&quotes, None, None).unwrap().curve;
    assert_eq!(base.days(), &days[..]);
    let qg = swap::quote_greeks(&quotes, &days, base.dfs(), &sw).unwrap();
    let h = 1e-6;
    for k in 0..quotes.len() {
        let mut pv = [0.0; 2];
        for (m, s) in [h, -h].into_iter().enumerate() {
            let mut qs = quotes.clone();
            match &mut qs[k] {
                Quote::Forward { rate, .. } | Quote::Swap { rate, .. } => *rate += s,
            }
            let c = bootstrap_dated(&qs, None, None).unwrap().curve;
            pv[m] = swap::pv_f64(c.days(), c.dfs(), &sw);
        }
        let bump = (pv[0] - pv[1]) / (2.0 * h);
        assert!(
            close(qg.d_quote[k], bump, 1e-6, 1e-4),
            "{k}: {} vs {bump}",
            qg.d_quote[k]
        );
    }
    assert!(swap::quote_greeks(&quotes[1..], &days, base.dfs(), &sw).is_none());
}
