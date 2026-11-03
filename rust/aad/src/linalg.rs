//! Dense solve for the implicit-function-theorem step of [`crate::swap`].

/// Solve `a x = b` (row-major `n x n`) by Gaussian elimination with partial
/// pivoting. Returns `None` for a singular matrix.
pub fn solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    assert_eq!(a.len(), n * n, "matrix must be n x n");
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().total_cmp(&a[j * n + c].abs()))?;
        if a[p * n + c] == 0.0 {
            return None;
        }
        if p != c {
            for j in 0..n {
                a.swap(p * n + j, c * n + j);
            }
            b.swap(p, c);
        }
        for i in c + 1..n {
            let f = a[i * n + c] / a[c * n + c];
            if f != 0.0 {
                for j in c..n {
                    a[i * n + j] -= f * a[c * n + j];
                }
                b[i] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|j| a[i * n + j] * x[j]).sum();
        x[i] = (b[i] - s) / a[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    #[test]
    fn solves_and_detects_singular() {
        let x = super::solve(vec![0.0, 2.0, 1.0, 1.0], vec![4.0, 3.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-15 && (x[1] - 2.0).abs() < 1e-15);
        assert!(super::solve(vec![1.0, 2.0, 2.0, 4.0], vec![1.0, 1.0]).is_none());
    }
}
