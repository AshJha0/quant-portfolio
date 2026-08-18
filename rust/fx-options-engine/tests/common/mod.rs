//! Shared assertion helper for the integration test suite.

/// Assert `|a - b| <= tol`, with a diagnostic message on failure.
pub fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{what}: |{a} - {b}| = {:.3e} > tol {tol:.1e}",
        (a - b).abs()
    );
}
