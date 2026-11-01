//! Property tests for qp-core: identities of the special functions, RNG
//! stream layout, the Cholesky kernel and thread-count independence.

use proptest::prelude::*;
use qp_core::linalg::cholesky_lower;
use qp_core::par::for_row_chunks;
use qp_core::rng::{Xoshiro256PlusPlus, Xoshiro256StarStar};
use qp_core::special::{
    acklam, erf, erf_c1_truncated, erfc, erfc_c1_truncated, halley_step, norm_cdf, norm_pdf,
};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// erfc(x) + erfc(-x) = 2 for both tables. Over |x| <= 6 the sum is
    /// formed from values at most 2, so the error is a few ulps of 2.
    #[test]
    fn erfc_reflection(x in -6.0f64..6.0) {
        prop_assert!((erfc(x) + erfc(-x) - 2.0).abs() <= 4.0 * f64::EPSILON);
        prop_assert!((erfc_c1_truncated(x) + erfc_c1_truncated(-x) - 2.0).abs() <= 4.0 * f64::EPSILON);
    }

    /// The two tables differ by a few ulps at most.
    #[test]
    fn tables_agree_to_a_few_ulps(x in -6.0f64..6.0) {
        let (a, b) = (erfc(x), erfc_c1_truncated(x));
        prop_assert!((a - b).abs() <= 4.0 * f64::EPSILON * a.abs());
        prop_assert!((erf(x) - erf_c1_truncated(x)).abs() <= 4.0 * f64::EPSILON);
    }

    /// erf is odd: exactly in the direct regime, to rounding beyond it.
    #[test]
    fn erf_is_odd(x in -6.0f64..6.0) {
        if x.abs() <= 0.46875 {
            prop_assert_eq!(erf(-x).to_bits(), (-erf(x)).to_bits());
        } else {
            prop_assert!((erf(-x) + erf(x)).abs() <= 2.0 * f64::EPSILON);
        }
    }

    /// The CDF is monotone and Phi(x) + Phi(-x) = 1.
    #[test]
    fn norm_cdf_symmetric_and_monotone(x in -8.0f64..8.0, dx in 1e-6f64..1.0) {
        prop_assert!(norm_cdf(x + dx) >= norm_cdf(x));
        prop_assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() <= 4.0 * f64::EPSILON);
    }

    /// Acklam plus two Halley steps inverts the CDF to near machine
    /// precision (Halley's cubic convergence from a 1e-9 start).
    #[test]
    fn acklam_halley_inverts(p in 1e-12f64..(1.0 - 1e-12)) {
        let mut x = acklam(p);
        for _ in 0..2 {
            x = halley_step(x, p, norm_cdf(x), norm_pdf(x));
        }
        let q = p.min(1.0 - p);
        prop_assert!((norm_cdf(x) - p).abs() <= 1e-13 * q.max(1e-300) + 4.0 * f64::EPSILON * p);
    }

    /// Same seed, same stream; fills equal repeated single draws.
    #[test]
    fn rng_stream_layout(seed in any::<u64>()) {
        let mut a = Xoshiro256PlusPlus::new(seed);
        let mut b = a.clone();
        let mut raw = [0u64; 17];
        a.fill_u64(&mut raw);
        for x in raw {
            prop_assert_eq!(x, b.next_u64());
        }
        let mut z = [0.0; 9];
        a.fill_standard_normal(&mut z);
        for v in z {
            prop_assert_eq!(v.to_bits(), b.standard_normal().to_bits());
        }
        let u = a.next_uniform();
        prop_assert!(u > 0.0 && u < 1.0);
        let mut s = Xoshiro256StarStar::new(seed);
        let v = s.next_open01();
        prop_assert!(v > 0.0 && v < 1.0);
    }

    /// L L^T reproduces A = B B^T + n I to rounding.
    #[test]
    fn cholesky_reconstructs(n in 1usize..7, entries in proptest::collection::vec(-1.0f64..1.0, 36)) {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = if i == j { n as f64 } else { 0.0 };
                for k in 0..n {
                    s += entries[i * 6 + k] * entries[j * 6 + k];
                }
                a[i * n + j] = s;
            }
        }
        let mut l = vec![0.0; n * n];
        cholesky_lower(&a, n, &mut l).unwrap();
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| l[i * n + k] * l[j * n + k]).sum();
                prop_assert!((s - a[i * n + j]).abs() <= 1e-12 * (2.0 * n as f64));
            }
        }
    }

    /// The result of a row-local kernel does not depend on the thread count.
    #[test]
    fn par_independent_of_threads(rows in 0usize..200, threads in 1usize..9, min_rows in 0usize..20) {
        let kernel = |first: usize, chunk: &mut [f64]| {
            for (r, row) in chunk.chunks_mut(3).enumerate() {
                let g = (first + r) as f64;
                for (c, v) in row.iter_mut().enumerate() {
                    *v = (g * 1.5 + c as f64).sin();
                }
            }
        };
        let mut one = vec![0.0; rows * 3];
        for_row_chunks(&mut one, 3, min_rows, 1, kernel);
        let mut many = vec![0.0; rows * 3];
        for_row_chunks(&mut many, 3, min_rows, threads, kernel);
        prop_assert_eq!(one, many);
    }

    /// Brent finds the root of a monotone cubic to the requested tolerance.
    #[test]
    fn brent_finds_root(r in -5.0f64..5.0) {
        let f = |x: f64| (x - r) * (1.0 + (x - r) * (x - r));
        let x = qp_core::roots::brentq(f, -10.0, 10.0, 1e-13, 200).unwrap();
        prop_assert!((x - r).abs() <= 1e-12);
    }
}
