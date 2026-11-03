//! Parity of the Rust credit-index and bond-futures kernels with the
//! Python reference values in `data/credit_bondfut_parity.txt` (written by
//! `python/equity/04-fixed-income/tools/gen_credit_bondfut_parity.py`).

use qp_core::bond_futures::{Bond, cme_cf, delivery_option, implied_repo, net_basis};
use qp_core::credit_index::{Schedule, isda_legs};

fn floats(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1e-300) || got == want,
        "{what}: {got} vs {want}"
    );
}

#[test]
fn matches_python() {
    let text = include_str!("data/credit_bondfut_parity.txt");
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .collect();
    let (mut nk, mut nr, mut nn, mut ni, mut no) = (0, 0, 0, 0, 0);
    let mut i = 0;
    while i < lines.len() {
        let p: Vec<&str> = lines[i].split_whitespace().collect();
        let x = floats(lines[i]);
        match p[0] {
            "K" => {
                let got = cme_cf(
                    x[0],
                    p[2].parse().unwrap(),
                    p[3].parse().unwrap(),
                    p[4] == "1",
                );
                close(got, x[4], 4.0 * f64::EPSILON, lines[i]);
                nk += 1;
                i += 1;
            }
            "R" => {
                close(
                    implied_repo(x[0], x[1], x[2], x[3], x[4]).unwrap(),
                    x[5],
                    4.0 * f64::EPSILON,
                    lines[i],
                );
                nr += 1;
                i += 1;
            }
            "N" => {
                let got = net_basis(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
                assert!((got - x[7]).abs() <= 1e-12, "{}: {got}", lines[i]);
                nn += 1;
                i += 1;
            }
            "I" => {
                let v: Vec<Vec<f64>> = (1..=8).map(|k| floats(lines[i + k])).collect();
                let s = Schedule {
                    acc_start: &v[4],
                    acc_end: &v[5],
                    pay_time: &v[6],
                    acc_frac: &v[7],
                };
                let (pr, rp, d) =
                    isda_legs(&v[0], &v[1], &v[2], &v[3], x[0], x[1], &s, x[2]).unwrap();
                assert!((pr - x[3]).abs() <= 1e-14, "prot {pr} vs {}", x[3]);
                close(rp, x[4], 1e-13, "rpv01");
                close(d, x[5], 1e-15, "df");
                ni += 1;
                i += 9;
            }
            "O" => {
                let nb = x[0] as usize;
                let rows: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..nb)
                    .map(|b| {
                        let j = i + 1 + 3 * b;
                        (floats(lines[j]), floats(lines[j + 1]), floats(lines[j + 2]))
                    })
                    .collect();
                let bonds: Vec<Bond<'_>> = rows
                    .iter()
                    .map(|(h, e, w)| Bond {
                        exps: e,
                        flows: w,
                        accrued: h[1],
                        freq: h[0] as u32,
                        fwd_yield: h[2],
                        cf: h[3],
                    })
                    .collect();
                let (fair, opt) = delivery_option(&bonds, x[1]).unwrap();
                close(fair, x[2], 1e-13, "fair");
                assert!((opt - x[3]).abs() <= 1e-11, "option {opt} vs {}", x[3]);
                no += 1;
                i += 1 + 3 * nb;
            }
            t => panic!("unknown tag {t}"),
        }
    }
    assert!(
        nk == 588 && nr == 3 && nn == 9 && ni == 12 && no == 3,
        "{nk} {nr} {nn} {ni} {no}"
    );
}
