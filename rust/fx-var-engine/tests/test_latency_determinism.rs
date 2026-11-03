//! Determinism and equivalence for the latency work.
//!
//! * Monte Carlo gives the same bits for any thread count and across runs.
//! * The pinned bits below were printed by the code before the parallel
//!   rewrite (commit d71e799), so they show the rewrite changed no result.
//! * The workspace kernels equal the allocating API bit for bit.

use fx_var_engine::prelude::*;

fn market() -> Market {
    Market::new(
        [("EUR", 1.1), ("JPY", 0.009), ("GBP", 1.27)],
        [("USD", 0.05), ("EUR", 0.03), ("JPY", 0.001), ("GBP", 0.04)],
    )
    .unwrap()
}

fn book() -> Book {
    Book::new(
        vec![
            Position::Spot(SpotPosition::new("EURUSD", 10e6, None)),
            Position::Forward(ForwardPosition::new("USDJPY", 5e6, 0.5, None)),
            Position::Spot(SpotPosition::new("GBPJPY", -3e6, None)),
        ],
        "EUR",
    )
    .unwrap()
}

fn returns() -> ReturnsMatrix {
    let f = CompiledBook::new(&book(), &market())
        .unwrap()
        .factors()
        .to_vec();
    let k = f.len();
    let mut d = Matrix::zeros(300, k);
    let mut r = Rng::new(5);
    for t in 0..300 {
        for j in 0..k {
            d.set(t, j, 0.005 * r.normal());
        }
    }
    ReturnsMatrix::new(f, d).unwrap()
}

fn opts(dist: McDist) -> MonteCarloOptions {
    let mut jumps = JumpSpec {
        prob: 0.05,
        ..Default::default()
    };
    jumps.mean.insert("FX:JPY".into(), -0.05);
    jumps.stdev.insert("FX:JPY".into(), 0.02);
    jumps.mean.insert("FX:GBP".into(), 0.03);
    MonteCarloOptions {
        n_scenarios: 30_000,
        seed: 9,
        dist,
        df: 5.0,
        jumps,
        horizon_days: 2.0,
        ..Default::default()
    }
}

fn bits(r: &MonteCarloResult) -> Vec<u64> {
    let mut v: Vec<u64> = r.pnl.iter().map(|x| x.to_bits()).collect();
    v.extend(
        [r.var, r.es, r.se_var, r.se_var_kde]
            .iter()
            .map(|x| x.to_bits()),
    );
    v
}

#[test]
fn monte_carlo_same_bits_for_any_thread_count() {
    let (b, m) = (book(), market());
    let cov = sample_cov(&returns()).unwrap();
    for dist in [McDist::Normal, McDist::StudentT, McDist::Jump] {
        let o = opts(dist);
        let reference = bits(&monte_carlo_var_with_threads(&b, &m, &cov, &o, 1).unwrap());
        for threads in [0, 2, 3, 7, 12, 64] {
            let got = bits(&monte_carlo_var_with_threads(&b, &m, &cov, &o, threads).unwrap());
            assert!(got == reference, "{dist:?} differs at {threads} threads");
        }
        let again = bits(&monte_carlo_var(&b, &m, &cov, &o).unwrap());
        assert!(again == reference, "{dist:?} differs between runs");
    }
}

// The pinned bits were recorded on x86_64 Linux (glibc libm, as in CI), and
// there they must match exactly. The Windows C runtime's exp and ln round
// differently in the last bit: measured on Windows 11 / MSVC target, ES moves
// by 1 ulp and the KDE standard error (which goes through exp) by 12 ulp, VaR
// and the plain standard error not at all. Elsewhere the test therefore checks
// a relative tolerance of 1e-12 (about 4500 ulp), which still catches any real
// change to the algorithm while allowing libm rounding.
#[test]
fn monte_carlo_matches_pre_rewrite_bits() {
    // (var, es, se_var, se_var_kde).to_bits() from the serial code.
    // Re-pinned in Round 8 (W20): the normals now come from the shared
    // qp_core::stats inverse CDF, whose Halley step uses a lower-tail-exact
    // normal CDF; every value moved by less than 1e-12 relative.
    // Re-pinned again in Round 8 (perf-fix lane, W4(b) for FX): scenario s
    // now draws from Philox stream (seed, s) with AS 241 normals and the
    // bootstrap from per-resample streams, so every value moved within Monte
    // Carlo error (VaR by 0.07 % normal, 1.2 % Student-t, 0.3 % jump).
    let pinned: [(McDist, [u64; 4]); 3] = [
        (
            McDist::Normal,
            [
                4685459142851392166,
                4686337053181062990,
                4656950455384134823,
                4655454433157111485,
            ],
        ),
        (
            McDist::StudentT,
            [
                4686233665241628561,
                4688493082336337432,
                4657742516503880608,
                4658038070050354006,
            ],
        ),
        (
            McDist::Jump,
            [
                4686013487656872710,
                4687018863430787878,
                4657157879210354217,
                4655846722963673680,
            ],
        ),
    ];
    let (b, m) = (book(), market());
    let cov = sample_cov(&returns()).unwrap();
    for (dist, want) in pinned {
        let r = monte_carlo_var(&b, &m, &cov, &opts(dist)).unwrap();
        let got = [
            r.var.to_bits(),
            r.es.to_bits(),
            r.se_var.to_bits(),
            r.se_var_kde.to_bits(),
        ];
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            assert_eq!(got, want, "{dist:?}");
        } else {
            for (g, w) in got.iter().zip(&want) {
                let (g, w) = (f64::from_bits(*g), f64::from_bits(*w));
                assert!(
                    (g - w).abs() <= 1e-12 * w.abs(),
                    "{dist:?}: got {g:e}, pinned {w:e}"
                );
            }
        }
    }
}

#[test]
fn workspace_tail_equals_allocating_tail() {
    let mut rng = Rng::new(17);
    for n in [1usize, 2, 7, 10, 99, 100, 101, 1000, 4097] {
        let pnl: Vec<f64> = (0..n)
            .map(|i| if i % 13 == 0 { 0.0 } else { 1e5 * rng.normal() })
            .collect();
        let mut ws = vec![f64::NAN; n + 3];
        for alpha in [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.4999] {
            let a = empirical_var_es_into(&pnl, alpha, &mut ws).unwrap();
            let b = empirical_var_es(&pnl, alpha, None).unwrap();
            assert_eq!(
                (a.0.to_bits(), a.1.to_bits()),
                (b.0.to_bits(), b.1.to_bits()),
                "n={n} alpha={alpha}"
            );
            let v = empirical_var_into(&pnl, alpha, &mut ws).unwrap();
            assert_eq!(
                v.to_bits(),
                empirical_var(&pnl, alpha, None).unwrap().to_bits()
            );
        }
    }
}

#[test]
fn workspace_pnl_equals_allocating_pnl() {
    let (b, m) = (book(), market());
    let cb = CompiledBook::new(&b, &m).unwrap();
    let rets = returns();
    let reference = cb.pnl_scenarios(&rets).unwrap();
    let mut out = vec![0.0; rets.n_obs()];
    cb.pnl_aligned_into(&rets.data, &mut out).unwrap();
    for (s, (a, r)) in out.iter().zip(&reference).enumerate() {
        assert_eq!(a.to_bits(), r.to_bits(), "row {s}");
        let row = cb.pnl_row(rets.data.row(s)).unwrap();
        assert_eq!(row.to_bits(), cb.pnl(rets.data.row(s)).unwrap().to_bits());
        assert_eq!(row.to_bits(), r.to_bits());
    }
}

#[test]
fn bootstrap_same_bits_for_any_thread_count() {
    // Large enough to take the parallel path, plus a small serial case.
    for (n, n_boot) in [(20_000usize, 120usize), (500, 30)] {
        let mut r = Rng::new(3);
        let pnl: Vec<f64> = (0..n).map(|_| 1e5 * r.normal()).collect();
        let reference =
            var_standard_error_bootstrap_with_threads(&pnl, 0.01, n_boot, 77, 1).unwrap();
        for threads in [0, 2, 3, 5, 7, 12, 64, 500] {
            let got =
                var_standard_error_bootstrap_with_threads(&pnl, 0.01, n_boot, 77, threads).unwrap();
            assert_eq!(
                got.to_bits(),
                reference.to_bits(),
                "n={n} threads={threads}"
            );
        }
        let default = var_standard_error_bootstrap(&pnl, 0.01, n_boot, 77).unwrap();
        assert_eq!(default.to_bits(), reference.to_bits());
    }
}
