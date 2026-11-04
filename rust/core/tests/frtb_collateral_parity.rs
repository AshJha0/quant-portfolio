//! Parity of the Rust FRTB and collateral kernels with the Python reference
//! values in `data/frtb_collateral_parity.txt` (written by
//! `python/equity/04-fixed-income/tools/gen_frtb_collateral_parity.py`). Bound:
//! 1e-12 relative to the largest entry of each output, the repository's
//! cross-language parity bound.

use std::collections::{BTreeMap, BTreeSet};

use qp_core::collateral::{
    Csa, Margin, collateral_balance, collateralised_exposure, lag_index, lagged_grid,
};
use qp_core::frtb::{
    BucketInput, Charges, CvrRow, across_buckets, aggregate, curvature_charge, net_cvr,
    within_bucket,
};

const FIXTURE: &str = include_str!("data/frtb_collateral_parity.txt");

fn rows() -> Vec<Vec<&'static str>> {
    FIXTURE
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.split_whitespace().collect())
        .collect()
}

fn floats(p: &[&str]) -> Vec<f64> {
    p.iter().map(|s| s.parse().unwrap()).collect()
}

fn close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-12 * scale,
            "{what}[{k}]: {g} vs {w} (scale {scale})"
        );
    }
}

fn charges(c: Charges) -> [f64; 4] {
    [c.medium, c.high, c.low, c.charge]
}

fn margin(r: &[&str]) -> Margin {
    let x = floats(&r[2..]);
    match r[1] {
        "S" => Margin::Scalar(x[0]),
        "T" => Margin::ByTime(x),
        "P" => Margin::ByPath(x),
        "F" => Margin::Full(x),
        k => panic!("unknown margin kind {k}"),
    }
}

#[test]
fn aggregate_matches_python() {
    let rows = rows();
    let mut cases = 0;
    let mut i = 0;
    while i < rows.len() {
        if rows[i][0] != "AGG" {
            i += 1;
            continue;
        }
        let nb: usize = rows[i][1].parse().unwrap();
        i += 1;
        let mut buckets = BTreeMap::new();
        for _ in 0..nb {
            let id: i32 = rows[i][1].parse().unwrap();
            let bucket = BucketInput {
                ws: floats(&rows[i + 1][1..]),
                rho: floats(&rows[i + 2][1..]),
                other: rows[i][2] == "1",
            };
            buckets.insert(id, bucket);
            i += 3;
        }
        let ids: Vec<i32> = buckets.keys().copied().collect();
        let gam = floats(&rows[i][1..]);
        let pos = |b: i32| ids.iter().position(|x| *x == b).unwrap();
        let got = aggregate(&buckets, |b, c| gam[pos(b) * nb + pos(c)]).unwrap();
        close(&charges(got), &floats(&rows[i + 1][1..]), "aggregate");
        i += 2;
        cases += 1;
    }
    assert_eq!(cases, 7);
}

#[test]
fn curvature_matches_python() {
    let rows = rows();
    let mut cases = 0;
    let mut i = 0;
    while i < rows.len() {
        if rows[i][0] != "CURV" {
            i += 1;
            continue;
        }
        let nb: usize = rows[i][1].parse().unwrap();
        i += 1;
        let mut buckets = BTreeMap::new();
        let mut other = BTreeSet::new();
        let mut rho: BTreeMap<i32, (Vec<&str>, Vec<f64>)> = BTreeMap::new();
        for _ in 0..nb {
            let id: i32 = rows[i][1].parse().unwrap();
            if rows[i][2] == "1" {
                other.insert(id);
            }
            let n: usize = rows[i][3].parse().unwrap();
            rho.insert(id, (rows[i + 1][1..].to_vec(), floats(&rows[i + 2][1..])));
            let list: Vec<CvrRow<&str>> = rows[i + 3..i + 3 + n]
                .iter()
                .map(|r| CvrRow {
                    key: r[1],
                    up: r[2].parse().unwrap(),
                    down: r[3].parse().unwrap(),
                })
                .collect();
            buckets.insert(id, list);
            i += 3 + n;
        }
        let ids: Vec<i32> = buckets.keys().copied().collect();
        let gam = floats(&rows[i][1..]);
        let pos = |b: i32| ids.iter().position(|x| *x == b).unwrap();
        let delta_rho = |b: i32, k: &&str, l: &&str| {
            let (keys, m) = &rho[&b];
            let at = |x: &&str| keys.iter().position(|y| y == x).unwrap();
            m[at(k) * keys.len() + at(l)]
        };
        let got = curvature_charge(
            &buckets,
            delta_rho,
            |b, c| gam[pos(b) * nb + pos(c)],
            &other,
        )
        .unwrap();
        close(&charges(got), &floats(&rows[i + 1][1..]), "curvature");
        i += 2;
        cases += 1;
    }
    assert_eq!(cases, 6);
}

#[test]
fn net_cvr_matches_python() {
    let mut cases = 0;
    for r in rows().iter().filter(|r| r[0] == "CVR") {
        let x = floats(&r[1..]);
        let (p, m) = net_cvr(x[0], x[1], x[2], x[3], x[4]);
        close(&[p, m], &x[5..7], "net_cvr");
        cases += 1;
    }
    assert_eq!(cases, 6);
}

#[test]
fn lagged_grid_matches_python() {
    let rows = rows();
    let mut cases = 0;
    for (i, r) in rows.iter().enumerate() {
        if r[0] == "LG" {
            let x = floats(&r[1..]);
            let got = lagged_grid(&x[1..], x[0]).unwrap();
            close(&got, &floats(&rows[i + 1][1..]), "lagged_grid");
            cases += 1;
        }
    }
    assert_eq!(cases, 5);
}

#[test]
fn collateral_matches_python() {
    let rows = rows();
    let mut cases = 0;
    for (i, r) in rows.iter().enumerate() {
        if r[0] != "COL" {
            continue;
        }
        let paths: usize = r[1].parse().unwrap();
        let n: usize = r[2].parse().unwrap();
        let t = floats(&rows[i + 1][1..]);
        let v = floats(&rows[i + 2][1..]);
        let c = floats(&rows[i + 3][1..]);
        let csa = Csa {
            threshold_cpty: c[0],
            threshold_bank: c[1],
            mta: c[2],
            mpor: c[3],
            initial: c[4],
            im_received: margin(&rows[i + 4]),
            im_posted: margin(&rows[i + 5]),
        };
        let bal = collateral_balance(&v, paths, n, &csa).unwrap();
        close(&bal, &floats(&rows[i + 6][1..]), "balance");
        let want: Vec<usize> = rows[i + 7][1..]
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(lag_index(&t, csa.mpor).unwrap(), want, "lag index");
        let e = collateralised_exposure(&t, &v, paths, &csa).unwrap();
        close(&e.collateral, &floats(&rows[i + 8][1..]), "collateral");
        close(&e.exposure, &floats(&rows[i + 9][1..]), "exposure");
        close(
            &e.negative_exposure,
            &floats(&rows[i + 10][1..]),
            "negative",
        );
        cases += 1;
    }
    assert_eq!(cases, 11);
}

#[test]
fn infinite_threshold_is_the_uncollateralised_limit() {
    let csa = Csa {
        threshold_cpty: f64::INFINITY,
        threshold_bank: f64::INFINITY,
        ..Csa::default()
    };
    let e = collateralised_exposure(&[0.0, 1.0], &[2.0, -3.0], 1, &csa).unwrap();
    assert_eq!(e.exposure, vec![2.0, 0.0]);
    assert_eq!(e.negative_exposure, vec![0.0, -3.0]);
}

#[test]
fn no_buckets_is_zero() {
    let none: BTreeMap<i32, BucketInput> = BTreeMap::new();
    assert_eq!(aggregate(&none, |_, _| 0.0).unwrap().charge, 0.0);
    let none: BTreeMap<i32, Vec<CvrRow<u8>>> = BTreeMap::new();
    let c = curvature_charge(&none, |_, _, _| 0.0, |_, _| 0.0, &BTreeSet::new()).unwrap();
    assert_eq!(c.charge, 0.0);
}

#[test]
fn other_bucket_does_not_read_rho() {
    let b = BucketInput {
        ws: vec![3.0, -4.0],
        rho: vec![],
        other: true,
    };
    let c = aggregate(&BTreeMap::from([(16, b)]), |_, _| 0.0).unwrap();
    assert_eq!(c.charge, 7.0);
}

#[test]
fn negative_k_does_not_panic_in_the_clip() {
    // numpy's clip accepts lower > upper; f64::clamp would panic.
    let g = [0.0, 1.0, 1.0, 0.0];
    assert!(across_buckets(&[-1.0, 0.0], &[5.0, -5.0], &g).is_ok());
}

#[test]
fn bad_input_is_refused() {
    assert!(within_bucket(&[1.0, 2.0], &[1.0]).is_err());
    assert!(within_bucket(&[f64::NAN], &[1.0]).is_err());
    assert!(across_buckets(&[1.0], &[1.0, 2.0], &[0.0]).is_err());
    let b = BucketInput {
        ws: vec![1.0],
        rho: vec![1.0],
        other: false,
    };
    let two = BTreeMap::from([(1, b.clone()), (2, b)]);
    assert!(aggregate(&two, |_, _| f64::NAN).is_err());
    let nan = BTreeMap::from([(
        1,
        vec![CvrRow {
            key: 0,
            up: f64::NAN,
            down: 0.0,
        }],
    )]);
    assert!(curvature_charge(&nan, |_, _, _| 0.0, |_, _| 0.0, &BTreeSet::new()).is_err());

    let neg = Csa {
        mta: -1.0,
        ..Csa::default()
    };
    assert!(collateral_balance(&[1.0], 1, 1, &neg).is_err());
    let nan = Csa {
        mpor: f64::NAN,
        ..Csa::default()
    };
    assert!(collateral_balance(&[1.0], 1, 1, &nan).is_err());
    let ok = Csa::default();
    assert!(collateral_balance(&[1.0], 2, 1, &ok).is_err());
    assert!(collateralised_exposure(&[], &[], 0, &ok).is_err());
    assert!(collateralised_exposure(&[1.0, 1.0], &[0.0, 0.0], 1, &ok).is_err());
    let im = Csa {
        im_received: Margin::Scalar(-1.0),
        ..Csa::default()
    };
    assert!(collateralised_exposure(&[0.0], &[0.0], 1, &im).is_err());
    let short = Csa {
        im_posted: Margin::ByTime(vec![1.0]),
        ..Csa::default()
    };
    assert!(collateralised_exposure(&[0.0, 1.0], &[0.0, 0.0], 1, &short).is_err());
    assert!(lagged_grid(&[], 0.1).is_err());
    assert!(lag_index(&[1.0, 0.0], 0.1).is_err());
}
