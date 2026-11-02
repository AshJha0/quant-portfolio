//! Cache-blocked matrix products that keep the naive bits.
//!
//! Tiles of `TILE` along each loop (Lam, Rothberg and Wolf, ASPLOS 1991).
//! The k (or row) tiles are visited in ascending order and every output
//! element is updated in place, so each `C(i, j)` sees the same sequence of
//! roundings `c = c + a(i, k) b(k, j)`, k ascending, as the textbook loop.
//! The result is the same `f64` (Rust never contracts `a * b + c` into an
//! FMA). Twin of `qp/gemm.hpp`.

/// Tile edge in elements.
pub const TILE: usize = 64;

/// `C += A B`, all row-major: A is `m x k`, B `k x n`, C `m x n`.
///
/// # Panics
/// If a slice is shorter than its shape.
pub fn gemm_blocked(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    for kb in (0..k).step_by(TILE) {
        let ke = (kb + TILE).min(k);
        for ib in (0..m).step_by(TILE) {
            let ie = (ib + TILE).min(m);
            for jb in (0..n).step_by(TILE) {
                let je = (jb + TILE).min(n);
                for i in ib..ie {
                    for p in kb..ke {
                        let aip = a[i * k + p];
                        let bp = &b[p * n + jb..p * n + je];
                        let ci = &mut c[i * n + jb..i * n + je];
                        for (cj, &bj) in ci.iter_mut().zip(bp) {
                            *cj += aip * bj;
                        }
                    }
                }
            }
        }
    }
}

/// Upper triangle (`j >= i`) of `C += D^T D`; D is `t x n`, C `n x n`,
/// row-major. Each `C(i, j)` adds `d(r, i) d(r, j)` for r ascending, the
/// order of the engines' row-streaming covariance loops.
///
/// # Panics
/// If a slice is shorter than its shape.
pub fn syrk_upper_blocked(d: &[f64], c: &mut [f64], t: usize, n: usize) {
    assert!(d.len() >= t * n && c.len() >= n * n);
    for rb in (0..t).step_by(TILE) {
        let re = (rb + TILE).min(t);
        for ib in (0..n).step_by(TILE) {
            let ie = (ib + TILE).min(n);
            for jb in (ib..n).step_by(TILE) {
                let je = (jb + TILE).min(n);
                for r in rb..re {
                    let dr = &d[r * n..r * n + n];
                    for i in ib..ie {
                        let di = dr[i];
                        let j0 = jb.max(i);
                        if j0 >= je {
                            continue;
                        }
                        let ci = &mut c[i * n + j0..i * n + je];
                        for (cj, &dj) in ci.iter_mut().zip(&dr[j0..je]) {
                            *cj += di * dj;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    fn rv(n: usize, seed: u64) -> Vec<f64> {
        let mut g = SplitMix64::new(seed);
        (0..n)
            .map(|_| (g.next_u64() >> 11) as f64 * 1e-12 - 4.5e3)
            .collect()
    }

    #[test]
    fn gemm_equals_naive_bits() {
        let (m, k, n) = (70, 131, 67);
        let (a, b) = (rv(m * k, 1), rv(k * n, 2));
        let mut c = vec![0.0; m * n];
        gemm_blocked(&a, &b, &mut c, m, k, n);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                assert_eq!(c[i * n + j].to_bits(), s.to_bits());
            }
        }
    }

    #[test]
    fn syrk_equals_row_streaming_bits() {
        let (t, n) = (300, 77);
        let d = rv(t * n, 3);
        let mut c = vec![0.0; n * n];
        let mut r = vec![0.0; n * n];
        syrk_upper_blocked(&d, &mut c, t, n);
        for row in 0..t {
            for i in 0..n {
                for j in i..n {
                    r[i * n + j] += d[row * n + i] * d[row * n + j];
                }
            }
        }
        assert_eq!(c, r);
    }
}
