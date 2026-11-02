//! Dense Cholesky factorisation on row-major slices.
//!
//! Both VaR engines keep their own `Matrix` type with its own validation
//! and error messages. They share this kernel, which does the arithmetic
//! in a fixed order so the factor is the same bit for bit in both.

/// A non-positive (or NaN) pivot met during factorisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NotPositiveDefinite {
    /// Row of the failing pivot.
    pub row: usize,
    /// The pivot value `a_ii - sum_k l_ik^2`.
    pub pivot: f64,
}

/// Lower-triangular Cholesky factor of the `n x n` row-major matrix `a`,
/// written into `l` (which the caller zeroes; only the lower triangle is
/// written).
///
/// Reads only the lower triangle of `a`. Fails on the first pivot that is
/// not `> 0`, which includes a NaN pivot: `!(s > 0.0)` is true for NaN,
/// where `s <= 0.0` would let a NaN through `sqrt` into an all-NaN factor.
///
/// # Panics
///
/// If `a` or `l` is shorter than `n * n`.
///
/// # Examples
///
/// ```
/// use qp_core::linalg::cholesky_lower;
/// let a = [4.0, 2.0, 2.0, 5.0];
/// let mut l = [0.0; 4];
/// cholesky_lower(&a, 2, &mut l).unwrap();
/// assert_eq!(l, [2.0, 0.0, 1.0, 2.0]);
/// let bad = [1.0, 2.0, 2.0, 1.0];
/// assert_eq!(cholesky_lower(&bad, 2, &mut [0.0; 4]).unwrap_err().row, 1);
/// ```
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn cholesky_lower(a: &[f64], n: usize, l: &mut [f64]) -> Result<(), NotPositiveDefinite> {
    assert!(
        a.len() >= n * n && l.len() >= n * n,
        "cholesky_lower: slice too short"
    );
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(s > 0.0) {
                    return Err(NotPositiveDefinite { row: i, pivot: s });
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Ok(())
}
