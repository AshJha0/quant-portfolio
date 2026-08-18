//! Shared assertion helpers for the integration test suite.

/// Assert `|a - b| <= tol`, with a labelled panic message.
#[macro_export]
macro_rules! assert_close {
    ($a:expr, $b:expr, $tol:expr) => {{
        let (a, b, tol): (f64, f64, f64) = ($a, $b, $tol);
        assert!(
            (a - b).abs() <= tol,
            "assert_close failed: {} vs {} (|diff| = {:.3e} > tol {:.1e}) [{} vs {}]",
            a,
            b,
            (a - b).abs(),
            tol,
            stringify!($a),
            stringify!($b)
        );
    }};
}

/// Assert relative closeness: `|a - b| <= tol * max(|a|, |b|, 1)`.
#[macro_export]
macro_rules! assert_rel_close {
    ($a:expr, $b:expr, $tol:expr) => {{
        let (a, b, tol): (f64, f64, f64) = ($a, $b, $tol);
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= tol * scale,
            "assert_rel_close failed: {} vs {} (rel diff = {:.3e} > tol {:.1e})",
            a,
            b,
            (a - b).abs() / scale,
            tol
        );
    }};
}
