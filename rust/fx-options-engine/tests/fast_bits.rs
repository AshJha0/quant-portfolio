//! The low-latency API returns the same bits as the classic API, the
//! batch pricers match the scalar ones, and Monte Carlo repeats exactly
//! for any thread count.

use fx_options_engine::fast::{
    binomial_price_ws, gk_price_batch, gk_price_k, greeks_batch, greeks_k, implied_vol_k,
    mc_price_par, mc_price_ws, BinomialWorkspace, GkBatch, GreeksBatchOut, KernelError,
    McWorkspace,
};
use fx_options_engine::rng::SplitMix64;
use fx_options_engine::{
    analytic_greeks, binomial_price, gk_price, implied_vol, mc_price, Exercise, GreeksResult,
    McResult, OptionType,
};

fn uni(r: &mut SplitMix64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
}

struct Cases {
    s: Vec<f64>,
    k: Vec<f64>,
    t: Vec<f64>,
    rd: Vec<f64>,
    rf: Vec<f64>,
    v: Vec<f64>,
    ty: Vec<OptionType>,
}

fn cases(n: usize, seed: u64) -> Cases {
    let mut r = SplitMix64::new(seed);
    let mut c = Cases {
        s: vec![],
        k: vec![],
        t: vec![],
        rd: vec![],
        rf: vec![],
        v: vec![],
        ty: vec![],
    };
    for i in 0..n {
        let s = uni(&mut r, 0.5, 150.0);
        c.s.push(s);
        c.k.push(s * uni(&mut r, 0.5, 1.6));
        c.t.push(uni(&mut r, 0.002, 5.0));
        c.rd.push(uni(&mut r, -0.01, 0.12));
        c.rf.push(uni(&mut r, -0.01, 0.12));
        c.v.push(uni(&mut r, 0.01, 0.8));
        c.ty.push(if i % 2 == 0 {
            OptionType::Call
        } else {
            OptionType::Put
        });
    }
    c
}

fn same_greeks(a: &GreeksResult, b: &GreeksResult) -> bool {
    [
        (a.price, b.price),
        (a.delta_spot, b.delta_spot),
        (a.delta_forward, b.delta_forward),
        (a.gamma, b.gamma),
        (a.vega, b.vega),
        (a.theta, b.theta),
        (a.rho_domestic, b.rho_domestic),
        (a.rho_foreign, b.rho_foreign),
        (a.vanna, b.vanna),
        (a.volga, b.volga),
    ]
    .iter()
    .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn same_mc(a: &McResult, b: &McResult) -> bool {
    a.price.to_bits() == b.price.to_bits()
        && a.std_error.to_bits() == b.std_error.to_bits()
        && a.ci_low.to_bits() == b.ci_low.to_bits()
        && a.ci_high.to_bits() == b.ci_high.to_bits()
        && a.n_paths == b.n_paths
        && a.method == b.method
}

#[test]
fn scalar_kernels_match_classic_api_bits() {
    let c = cases(4000, 11);
    for i in 0..c.s.len() {
        let (s, k, t, rd, rf, v, ty) = (c.s[i], c.k[i], c.t[i], c.rd[i], c.rf[i], c.v[i], c.ty[i]);
        let a = gk_price(s, k, t, rd, rf, v, ty).unwrap();
        assert_eq!(
            a.to_bits(),
            gk_price_k(s, k, t, rd, rf, v, ty).unwrap().to_bits(),
            "case {i}"
        );
        let g = analytic_greeks(s, k, t, rd, rf, v, ty).unwrap();
        assert!(
            same_greeks(&g, &greeks_k(s, k, t, rd, rf, v, ty).unwrap()),
            "case {i}"
        );
        let iv_a = implied_vol(a, s, k, t, rd, rf, ty);
        let iv_b = implied_vol_k(a, s, k, t, rd, rf, ty);
        match (iv_a, iv_b) {
            (Ok(x), Ok(y)) => assert_eq!(x.to_bits(), y.to_bits(), "case {i}"),
            (Err(_), Err(_)) => {}
            (x, y) => panic!("case {i}: classic {x:?} vs kernel {y:?}"),
        }
    }
    // Limits: t = 0 and sigma = 0 follow the classic branches.
    for ty in [OptionType::Call, OptionType::Put] {
        for (t, v) in [(0.0, 0.1), (0.5, 0.0), (0.0, 0.0)] {
            let a = gk_price(1.1, 1.05, t, 0.04, 0.03, v, ty).unwrap();
            let b = gk_price_k(1.1, 1.05, t, 0.04, 0.03, v, ty).unwrap();
            assert_eq!(a.to_bits(), b.to_bits());
            assert_eq!(
                analytic_greeks(1.1, 1.05, t, 0.04, 0.03, v, ty).is_err(),
                greeks_k(1.1, 1.05, t, 0.04, 0.03, v, ty).is_err()
            );
        }
    }
    // Invalid inputs fail on both APIs.
    for bad in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert!(gk_price(bad, 1.0, 1.0, 0.0, 0.0, 0.1, OptionType::Call).is_err());
        assert!(gk_price_k(bad, 1.0, 1.0, 0.0, 0.0, 0.1, OptionType::Call).is_err());
    }
}

#[test]
fn batch_matches_scalar_bits() {
    let c = cases(1031, 23);
    let b = GkBatch {
        s: &c.s,
        k: &c.k,
        t: &c.t,
        r_d: &c.rd,
        r_f: &c.rf,
        sigma: &c.v,
        option_type: &c.ty,
    };
    let n = c.s.len();
    let mut out = vec![0.0; n];
    gk_price_batch(&b, &mut out).unwrap();
    let mut g: Vec<Vec<f64>> = vec![vec![0.0; n]; 10];
    {
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] = &mut g[..] else {
            panic!("ten columns")
        };
        let mut go = GreeksBatchOut {
            price: a0,
            delta_spot: a1,
            delta_forward: a2,
            gamma: a3,
            vega: a4,
            theta: a5,
            rho_domestic: a6,
            rho_foreign: a7,
            vanna: a8,
            volga: a9,
        };
        greeks_batch(&b, &mut go).unwrap();
    }
    for i in 0..n {
        let p = gk_price(c.s[i], c.k[i], c.t[i], c.rd[i], c.rf[i], c.v[i], c.ty[i]).unwrap();
        assert_eq!(p.to_bits(), out[i].to_bits(), "price {i}");
        let r = analytic_greeks(c.s[i], c.k[i], c.t[i], c.rd[i], c.rf[i], c.v[i], c.ty[i]).unwrap();
        let fromb = GreeksResult {
            price: g[0][i],
            delta_spot: g[1][i],
            delta_forward: g[2][i],
            gamma: g[3][i],
            vega: g[4][i],
            theta: g[5][i],
            rho_domestic: g[6][i],
            rho_foreign: g[7][i],
            vanna: g[8][i],
            volga: g[9][i],
        };
        assert!(same_greeks(&r, &fromb), "greeks {i}");
    }
}

#[test]
fn batch_reports_first_bad_index_and_lengths() {
    let mut c = cases(16, 5);
    c.v[9] = -0.1;
    c.s[12] = f64::NAN;
    let b = GkBatch {
        s: &c.s,
        k: &c.k,
        t: &c.t,
        r_d: &c.rd,
        r_f: &c.rf,
        sigma: &c.v,
        option_type: &c.ty,
    };
    let mut out = vec![0.0; 16];
    let e = gk_price_batch(&b, &mut out).unwrap_err();
    assert_eq!((e.index, e.error), (9, KernelError::NegativeVol));
    let mut short = vec![0.0; 15];
    assert_eq!(
        gk_price_batch(&b, &mut short).unwrap_err().error,
        KernelError::LengthMismatch
    );
}

#[test]
fn binomial_workspace_matches_classic_bits() {
    let mut ws = BinomialWorkspace::new();
    let c = cases(60, 3);
    for i in 0..c.s.len() {
        for ex in [Exercise::European, Exercise::American] {
            for steps in [1u32, 2, 7, 64, 301] {
                let (s, k, t, rd, rf, v, ty) =
                    (c.s[i], c.k[i], c.t[i], c.rd[i], c.rf[i], c.v[i], c.ty[i]);
                let a = binomial_price(s, k, t, rd, rf, v, ty, steps, ex);
                let b = binomial_price_ws(&mut ws, s, k, t, rd, rf, v, ty, steps, ex);
                match (a, b) {
                    (Ok(x), Ok(y)) => {
                        assert_eq!(x.to_bits(), y.to_bits(), "case {i} steps {steps}")
                    }
                    (Err(_), Err(_)) => {}
                    (x, y) => panic!("case {i} steps {steps}: {x:?} vs {y:?}"),
                }
            }
        }
    }
    for ex in [Exercise::European, Exercise::American] {
        let a = binomial_price(1.1, 1.2, 1.0, 0.05, 0.01, 0.0, OptionType::Put, 50, ex).unwrap();
        let b = binomial_price_ws(
            &mut ws,
            1.1,
            1.2,
            1.0,
            0.05,
            0.01,
            0.0,
            OptionType::Put,
            50,
            ex,
        )
        .unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn monte_carlo_matches_classic_for_every_thread_count() {
    let mut ws = McWorkspace::new();
    for (anti, cv) in [(true, true), (true, false), (false, true), (false, false)] {
        for n_paths in [3u64, 4, 5, 1001, 20_000] {
            for ty in [OptionType::Call, OptionType::Put] {
                let a = mc_price(
                    1.10, 1.12, 0.5, 0.0425, 0.029, 0.0925, ty, n_paths, 42, anti, cv,
                );
                let b = mc_price_ws(
                    &mut ws, 1.10, 1.12, 0.5, 0.0425, 0.029, 0.0925, ty, n_paths, 42, anti, cv,
                );
                let a = match (a, b) {
                    (Ok(x), Ok(y)) => {
                        assert!(same_mc(&x, &y), "{anti} {cv} {n_paths}");
                        x
                    }
                    (Err(_), Err(_)) => continue,
                    (x, y) => panic!("{x:?} vs {y:?}"),
                };
                for threads in [2usize, 3, 4, 7, 12, 64] {
                    let p = mc_price_par(
                        &mut ws, 1.10, 1.12, 0.5, 0.0425, 0.029, 0.0925, ty, n_paths, 42, anti, cv,
                        threads,
                    )
                    .unwrap();
                    assert!(same_mc(&a, &p), "{anti} {cv} {n_paths} threads {threads}");
                }
            }
        }
    }
}

#[test]
fn monte_carlo_repeats_across_runs() {
    let run = |threads| {
        let mut ws = McWorkspace::new();
        mc_price_par(
            &mut ws,
            1.10,
            1.05,
            1.0,
            0.03,
            0.01,
            0.12,
            OptionType::Call,
            200_001,
            9,
            true,
            true,
            threads,
        )
        .unwrap()
    };
    let first = run(4);
    for threads in [1, 4, 4, 8] {
        assert!(same_mc(&first, &run(threads)));
    }
    let mut ws = McWorkspace::new();
    assert_eq!(
        mc_price_par(
            &mut ws,
            1.1,
            1.1,
            1.0,
            0.0,
            0.0,
            0.1,
            OptionType::Call,
            100,
            1,
            true,
            true,
            0
        )
        .unwrap_err(),
        KernelError::Threads
    );
    let other = mc_price_par(
        &mut ws,
        1.10,
        1.05,
        1.0,
        0.03,
        0.01,
        0.12,
        OptionType::Call,
        200_001,
        10,
        true,
        true,
        4,
    )
    .unwrap();
    assert_ne!(first.price.to_bits(), other.price.to_bits());
}
