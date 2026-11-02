//! The fast API, the batch pricers and the workspaces return the same
//! bits as the classic API, and Monte Carlo is reproducible across runs
//! and thread counts.

use eq_options_engine::fast::{
    bs_greeks_batch, bs_price_batch, try_black76_greeks, try_black76_price, try_bs_greeks,
    try_bs_price, try_crr_price, try_implied_vol, try_mc_price, BatchError, BsBatch, ErrorCode,
    GreeksOut,
};
use eq_options_engine::{
    black76_greeks, black76_price, bs_greeks, bs_price, crr_price, implied_vol, mc_price, BsGreeks,
    CrrWorkspace, Exercise, McWorkspace, OptionType,
};

/// Small xorshift for varied inputs (test-only, fixed seed).
struct Gen(u64);

impl Gen {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / 9_007_199_254_740_992.0
    }
}

#[derive(Clone, Copy)]
struct Row {
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    v: f64,
    q: f64,
    ot: OptionType,
}

/// Varied rows, including expiry, zero vol, zero spot and invalid inputs.
fn rows(n: usize) -> Vec<Row> {
    let mut g = Gen(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|i| {
            let mut row = Row {
                s: 1.0 + 300.0 * g.unit(),
                k: 1.0 + 300.0 * g.unit(),
                t: 0.002 + 5.0 * g.unit(),
                r: -0.03 + 0.15 * g.unit(),
                v: 0.01 + 2.0 * g.unit(),
                q: 0.1 * g.unit(),
                ot: if i % 2 == 0 {
                    OptionType::Call
                } else {
                    OptionType::Put
                },
            };
            match i % 41 {
                0 => row.t = 0.0,
                1 => row.v = 0.0,
                2 => row.s = 0.0,
                3 => row.k = 0.0,
                4 => row.s = -1.0,
                5 => row.v = f64::NAN,
                6 => row.r = f64::INFINITY,
                _ => {}
            }
            row
        })
        .collect()
}

fn greek_bits(g: &BsGreeks) -> [u64; 8] {
    [
        g.price, g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga,
    ]
    .map(f64::to_bits)
}

#[test]
fn try_api_matches_classic_bit_for_bit() {
    for x in rows(20_000) {
        let a = bs_price(x.s, x.k, x.t, x.r, x.v, x.q, x.ot)
            .ok()
            .map(f64::to_bits);
        let b = try_bs_price(x.s, x.k, x.t, x.r, x.v, x.q, x.ot)
            .ok()
            .map(f64::to_bits);
        assert_eq!(a, b);

        let a = bs_greeks(x.s, x.k, x.t, x.r, x.v, x.q, x.ot)
            .ok()
            .map(|g| greek_bits(&g));
        let b = try_bs_greeks(x.s, x.k, x.t, x.r, x.v, x.q, x.ot)
            .ok()
            .map(|g| greek_bits(&g));
        assert_eq!(a, b);

        let a = black76_price(x.s, x.k, x.t, x.r, x.v, x.ot)
            .ok()
            .map(f64::to_bits);
        let b = try_black76_price(x.s, x.k, x.t, x.r, x.v, x.ot)
            .ok()
            .map(f64::to_bits);
        assert_eq!(a, b);

        let a = black76_greeks(x.s, x.k, x.t, x.r, x.v, x.ot)
            .ok()
            .map(|g| g.delta.to_bits());
        let b = try_black76_greeks(x.s, x.k, x.t, x.r, x.v, x.ot)
            .ok()
            .map(|g| g.delta.to_bits());
        assert_eq!(a, b);

        if let Ok(p) = bs_price(x.s, x.k, x.t, x.r, x.v, x.q, x.ot) {
            for quote in [p, 0.5 * p, 1.5 * p] {
                let a = implied_vol(quote, x.s, x.k, x.t, x.r, x.q, x.ot);
                let b = try_implied_vol(quote, x.s, x.k, x.t, x.r, x.q, x.ot);
                assert_eq!(
                    a.as_ref().ok().map(|v| v.to_bits()),
                    b.ok().map(f64::to_bits)
                );
                if let Err(e) = b {
                    // Same error category as the classic API.
                    let classic: eq_options_engine::PricingError = e.into();
                    assert_eq!(
                        std::mem::discriminant(&classic),
                        std::mem::discriminant(a.as_ref().unwrap_err())
                    );
                }
            }
        }
    }
}

#[test]
fn batch_matches_scalar_bit_for_bit() {
    let rs = rows(5_003);
    // Valid rows only for the full-length run; invalid rows are tested
    // for the first-failure index below.
    let ok: Vec<Row> = rs
        .iter()
        .copied()
        .filter(|x| try_bs_greeks(x.s, x.k, x.t, x.r, x.v, x.q, x.ot).is_ok())
        .collect();
    let col = |f: fn(&Row) -> f64| ok.iter().map(f).collect::<Vec<f64>>();
    let (s, k, t, r, v, q) = (
        col(|x| x.s),
        col(|x| x.k),
        col(|x| x.t),
        col(|x| x.r),
        col(|x| x.v),
        col(|x| x.q),
    );
    let ot: Vec<OptionType> = ok.iter().map(|x| x.ot).collect();
    let b = BsBatch {
        s: &s,
        k: &k,
        t: &t,
        r: &r,
        sigma: &v,
        q: &q,
        option_type: &ot,
    };
    let n = ok.len();

    let mut prices = vec![0.0; n];
    bs_price_batch(&b, &mut prices).unwrap();
    let mut cols = vec![vec![0.0; n]; 8];
    let [c0, c1, c2, c3, c4, c5, c6, c7] = &mut cols[..] else {
        unreachable!()
    };
    let mut out = GreeksOut {
        price: c0,
        delta: c1,
        gamma: c2,
        vega: c3,
        theta: c4,
        rho: c5,
        vanna: c6,
        volga: c7,
    };
    bs_greeks_batch(&b, &mut out).unwrap();
    for (i, x) in ok.iter().enumerate() {
        let p = bs_price(x.s, x.k, x.t, x.r, x.v, x.q, x.ot).unwrap();
        assert_eq!(prices[i].to_bits(), p.to_bits(), "row {i}");
        let g = greek_bits(&bs_greeks(x.s, x.k, x.t, x.r, x.v, x.q, x.ot).unwrap());
        let got: Vec<u64> = cols.iter().map(|c| c[i].to_bits()).collect();
        assert_eq!(got, g, "row {i}");
    }

    // First failing row is reported; earlier rows are written.
    let mut bad_s = s.clone();
    bad_s[7] = f64::NAN;
    let bb = BsBatch { s: &bad_s, ..b };
    let mut out2 = vec![0.0; n];
    assert_eq!(
        bs_price_batch(&bb, &mut out2),
        Err(BatchError {
            index: 7,
            code: ErrorCode::NanInput
        })
    );
    assert_eq!(out2[6].to_bits(), prices[6].to_bits());

    // Length mismatch is an error, not a panic.
    let mut short = vec![0.0; n - 1];
    assert_eq!(
        bs_price_batch(&b, &mut short).map_err(|e| e.code),
        Err(ErrorCode::LengthMismatch)
    );
}

#[test]
fn crr_workspace_matches_classic_bit_for_bit() {
    let mut ws = CrrWorkspace::new();
    for (i, x) in rows(1_500).into_iter().enumerate() {
        let n = 1 + (i * 7) % 400;
        for ex in [Exercise::European, Exercise::American] {
            let a = crr_price(x.s, x.k, x.t, x.r, x.v, x.q, x.ot, ex, n)
                .ok()
                .map(f64::to_bits);
            let b = try_crr_price(x.s, x.k, x.t, x.r, x.v, x.q, x.ot, ex, n, &mut ws)
                .ok()
                .map(f64::to_bits);
            assert_eq!(a, b, "row {i} {ex:?} n={n}");
        }
    }
    let x = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, OptionType::Put);
    assert_eq!(
        try_crr_price(
            x.0,
            x.1,
            x.2,
            x.3,
            x.4,
            x.5,
            x.6,
            Exercise::American,
            0,
            &mut ws
        ),
        Err(ErrorCode::StepsOutOfRange)
    );
}

#[test]
fn mc_workspace_matches_classic_bit_for_bit() {
    let mut ws = McWorkspace::new();
    for (i, x) in rows(240).into_iter().enumerate() {
        let n_paths = 2 + 37 * i;
        for (anti, cv) in [(false, false), (true, false), (false, true), (true, true)] {
            let a = mc_price(
                x.s, x.k, x.t, x.r, x.v, x.q, x.ot, n_paths, anti, cv, i as u64,
            )
            .ok()
            .map(|m| [m.price, m.std_error, m.ci_low, m.ci_high].map(f64::to_bits));
            let b = try_mc_price(
                x.s, x.k, x.t, x.r, x.v, x.q, x.ot, n_paths, anti, cv, i as u64, 1, &mut ws,
            )
            .ok()
            .map(|m| [m.price, m.std_error, m.ci_low, m.ci_high].map(f64::to_bits));
            assert_eq!(a, b, "row {i} n={n_paths} anti={anti} cv={cv}");
        }
    }
}

#[test]
fn mc_is_deterministic_across_runs_and_thread_counts() {
    let args = (100.0, 95.0, 0.8, 0.04, 0.3, 0.01, OptionType::Call);
    for n_paths in [3, 4, 101, 1_000, 100_001] {
        for (anti, cv) in [(false, false), (true, false), (false, true), (true, true)] {
            let reference = mc_price(
                args.0, args.1, args.2, args.3, args.4, args.5, args.6, n_paths, anti, cv, 2024,
            );
            let reference = match reference {
                Ok(m) => m,
                Err(_) => continue, // too few samples for this layout
            };
            let mut ws = McWorkspace::new();
            for run in 0..2 {
                for threads in [1, 2, 3, 4, 7, 12] {
                    let m = try_mc_price(
                        args.0, args.1, args.2, args.3, args.4, args.5, args.6, n_paths, anti, cv,
                        2024, threads, &mut ws,
                    )
                    .unwrap();
                    assert_eq!(
                        (m.price.to_bits(), m.std_error.to_bits(), m.n_paths),
                        (
                            reference.price.to_bits(),
                            reference.std_error.to_bits(),
                            reference.n_paths
                        ),
                        "run {run} n={n_paths} anti={anti} cv={cv} threads={threads}"
                    );
                }
            }
        }
    }
}

/// `threads == 0` is an error, as in the C++ engine (`Errc::BadThreads`),
/// for every input including the `T == 0` shortcut; it used to run on one
/// thread. A bad path count is reported first, in the C++ order.
#[test]
fn mc_rejects_zero_threads() {
    use eq_options_engine::fast::ErrorCode;
    let mut ws = McWorkspace::new();
    for t in [0.8, 0.0] {
        let e = try_mc_price(
            100.0,
            95.0,
            t,
            0.04,
            0.3,
            0.01,
            OptionType::Call,
            1_000,
            true,
            true,
            7,
            0,
            &mut ws,
        )
        .unwrap_err();
        assert_eq!(e, ErrorCode::BadThreads);
        assert_eq!(e.to_string(), "threads must be at least 1");
    }
    let e = try_mc_price(
        100.0,
        95.0,
        0.8,
        0.04,
        0.3,
        0.01,
        OptionType::Call,
        1,
        false,
        false,
        7,
        0,
        &mut ws,
    )
    .unwrap_err();
    assert_eq!(e, ErrorCode::PathsOutOfRange);
}
