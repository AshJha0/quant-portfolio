//! The Rust bootstrap against QuantLib 1.44 through the shared fixture
//! `python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt`
//! (the same file the Python and C++ tests read). Pillar discount factors
//! to 1e-12; every quote reprices to 1e-10 in rate.

use qp_rates_curves::*;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Default)]
struct Block {
    quotes: Vec<Quote>,
    xbs: Vec<XccyBasisQuote>,
    expect: Vec<(i64, f64)>,
    nodes: Vec<(i64, f64)>,
}

struct Fixture {
    blocks: HashMap<String, Block>,
    hwconv: Vec<[f64; 6]>,
}

fn fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt");
    let text = std::fs::read_to_string(&path).expect("fixture readable");
    let rows: Vec<Vec<String>> = text
        .lines()
        .map(|l| {
            l.split('#')
                .next()
                .unwrap()
                .split_whitespace()
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .filter(|r: &Vec<String>| !r.is_empty())
        .collect();
    let mut fx = Fixture {
        blocks: HashMap::new(),
        hwconv: Vec::new(),
    };
    let mut cur = String::new();
    let mut i = 0;
    let fl = |r: &Vec<String>| FloatPeriod {
        start: r[1].parse().unwrap(),
        end: r[2].parse().unwrap(),
        pay: r[3].parse().unwrap(),
        accrual: r[4].parse().unwrap(),
    };
    while i < rows.len() {
        let r = &rows[i];
        i += 1;
        let num = |k: usize| -> f64 { r[k].parse().unwrap() };
        let int = |k: usize| -> i64 { r[k].parse().unwrap() };
        match r[0].as_str() {
            "build" | "given" => {
                cur = r[1].clone();
                fx.blocks.insert(cur.clone(), Block::default());
            }
            "end" => {}
            "node" => fx
                .blocks
                .get_mut(&cur)
                .unwrap()
                .nodes
                .push((int(1), num(2))),
            "expect" => fx
                .blocks
                .get_mut(&cur)
                .unwrap()
                .expect
                .push((int(1), num(2))),
            "fwd" => fx
                .blocks
                .get_mut(&cur)
                .unwrap()
                .quotes
                .push(Quote::Forward {
                    start: int(1),
                    end: int(2),
                    accrual: num(3),
                    rate: num(4),
                }),
            "swap" => {
                let (nf, nl) = (int(2) as usize, int(3) as usize);
                let fixed = rows[i..i + nf]
                    .iter()
                    .map(|a| FixedPeriod {
                        pay: a[1].parse().unwrap(),
                        accrual: a[2].parse().unwrap(),
                    })
                    .collect();
                let floating = rows[i + nf..i + nf + nl].iter().map(fl).collect();
                i += nf + nl;
                fx.blocks.get_mut(&cur).unwrap().quotes.push(Quote::Swap {
                    rate: num(1),
                    fixed,
                    floating,
                });
            }
            "xbs" => {
                let (nf, nd) = (int(4) as usize, int(5) as usize);
                let foreign = rows[i..i + nf].iter().map(fl).collect();
                let domestic = rows[i + nf..i + nf + nd].iter().map(fl).collect();
                i += nf + nd;
                fx.blocks.get_mut(&cur).unwrap().xbs.push(XccyBasisQuote {
                    basis: num(1),
                    start: int(2),
                    end: int(3),
                    foreign,
                    domestic,
                });
            }
            "hwconv" => fx
                .hwconv
                .push([num(1), num(2), num(3), num(4), num(5), num(6)]),
            other => panic!("unknown fixture line {other}"),
        }
    }
    fx
}

fn curve_of(pairs: &[(i64, f64)]) -> DatedCurve {
    DatedCurve::new(
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    )
    .unwrap()
}

fn check(res: &BootstrapResult, expect: &[(i64, f64)]) {
    assert_eq!(
        res.curve.days(),
        expect.iter().map(|p| p.0).collect::<Vec<_>>().as_slice()
    );
    for (&got, &(_, want)) in res.curve.dfs().iter().zip(expect) {
        assert!((got - want).abs() <= 1e-12, "df {got} vs QuantLib {want}");
    }
    assert!(
        res.max_abs_residual() < 1e-10,
        "residual {}",
        res.max_abs_residual()
    );
}

#[test]
fn ois_curve_matches_quantlib() {
    let fx = fixture();
    let b = &fx.blocks["usd_ois"];
    let res = bootstrap_dated(&b.quotes, None, None).unwrap();
    assert_eq!(res.diagnostics.len(), 18);
    check(&res, &b.expect);
}

#[test]
fn projection_curve_matches_quantlib() {
    let fx = fixture();
    let ois = bootstrap_dated(&fx.blocks["usd_ois"].quotes, None, None)
        .unwrap()
        .curve;
    let b = &fx.blocks["term3m"];
    let res = bootstrap_dated(&b.quotes, Some(&ois), None).unwrap();
    check(&res, &b.expect);
}

#[test]
fn xccy_curve_matches_quantlib() {
    let fx = fixture();
    let usd = curve_of(&fx.blocks["usd_ois"].expect);
    let eur = curve_of(&fx.blocks["eur_ois"].nodes);
    let b = &fx.blocks["eur_xccy"];
    let res = bootstrap_xccy_discount(&b.xbs, &eur, &usd).unwrap();
    assert_eq!(res.curve.days().len(), 6);
    check(&res, &b.expect);
}

#[test]
fn hull_white_convexity_matches_quantlib() {
    let fx = fixture();
    assert_eq!(fx.hwconv.len(), 3);
    for h in &fx.hwconv {
        let got = hull_white_convexity(h[0], h[1], h[2], h[3], h[4]).unwrap();
        assert!(((got - h[5]) / h[5]).abs() < 1e-13, "{got} vs {}", h[5]);
    }
    assert!(hull_white_convexity(96.0, 2.0, 1.0, 0.01, 0.03).is_err());
}

#[test]
fn flat_forward_extrapolation_and_errors() {
    let fx = fixture();
    let c = bootstrap_dated(&fx.blocks["usd_ois"].quotes, None, None)
        .unwrap()
        .curve;
    let last = *c.days().last().unwrap() as f64;
    assert!((c.forward_rate(last - 10.0) - c.forward_rate(last + 1000.0)).abs() < 1e-14);
    assert!(DatedCurve::new(vec![5, 5], vec![0.9, 0.8]).is_err());
    assert!(DatedCurve::new(vec![5], vec![-0.9]).is_err());
    assert!(c.log_df(-1.0).is_nan());
    assert!(bootstrap_dated(&[], None, None).is_err());
    let q = Quote::Forward {
        start: 0,
        end: 30,
        accrual: 30.0 / 360.0,
        rate: 0.05,
    };
    let q2 = Quote::Forward {
        start: 10,
        end: 30,
        accrual: 20.0 / 360.0,
        rate: 0.05,
    };
    assert!(
        bootstrap_dated(&[q.clone(), q2], None, None)
            .unwrap_err()
            .0
            .contains("share pillar")
    );
    let crazy = Quote::Forward {
        start: 0,
        end: 30,
        accrual: 30.0 / 360.0,
        rate: -50.0,
    };
    assert!(bootstrap_dated(&[crazy], None, None).is_err());
    assert!(bootstrap_dated(&[q], Some(&c), Some(&c)).is_err());
    assert!((futures_rate(95.0, 1e-4) - 0.0499).abs() < 1e-15);
}
