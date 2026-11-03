//! Fuzz properties for qp-core, shared by the cargo-fuzz targets and the
//! stable corpus-replay test.
//!
//! Each `check_*` function takes the fuzzer's raw bytes, maps them to
//! bounded inputs with [`FuzzInput`], and panics when a property fails. The
//! properties match the C++ targets in `cpp/core/fuzz/` one for one, so a
//! case that breaks one language can be replayed in the other.

use qp_core::iv::{
    Cody, black_implied_vol, normalised_otm_call, normalised_otm_call_complement, total_vol,
};
use qp_core::linalg::cholesky_lower;

/// Bytes to bounded doubles, 8 bytes per value, zero-padded at the end.
/// Same mapping as `cpp/core/fuzz/fuzz_input.hpp`.
pub struct FuzzInput<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> FuzzInput<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        if self.pos < self.data.len() {
            let take = (self.data.len() - self.pos).min(8);
            b[..take].copy_from_slice(&self.data[self.pos..self.pos + take]);
        }
        self.pos += 8;
        u64::from_le_bytes(b)
    }

    pub fn in_range(&mut self, lo: f64, hi: f64) -> f64 {
        let f = (self.u64() >> 11) as f64 * f64::powi(2.0, -53);
        lo + (hi - lo) * f
    }

    pub fn flag(&mut self) -> bool {
        self.u64() & 1 != 0
    }
}

/// Implied vol: (1) any quote in a wide box returns Ok with a finite
/// positive vol or an Err, never a panic; (2) a normalised OTM price built
/// from (x, s) round-trips: the vol to 1e-10 relative or the price to 1e-9
/// relative (deep OTM the price elasticity is about x^2 / s^2, so only one
/// of the two can be asked for everywhere).
pub fn check_implied_vol(data: &[u8]) {
    let mut fin = FuzzInput::new(data);
    let f = fin.in_range(1e-6, 1e6);
    let k = fin.in_range(1e-6, 1e6);
    let t = fin.in_range(1e-6, 50.0);
    let df = fin.in_range(1e-3, 1.0);
    let is_call = fin.flag();
    let upper = if is_call { df * f } else { df * k };
    let price = fin.in_range(0.0, upper);
    let mut evals = 0u32;
    if let Ok(v) = black_implied_vol::<Cody>(price, f, k, t, df, is_call, &mut evals) {
        assert!(v.is_finite() && v > 0.0, "vol {v} for price {price}");
    }

    let x = -fin.in_range(0.0, 10.0);
    let s = fin.in_range(1e-3, 6.0);
    let beta = normalised_otm_call::<Cody>(x, s);
    let u = normalised_otm_call_complement::<Cody>(x, s);
    if beta > 1e-280 && u > 1e-280 && beta.is_finite() {
        let mut evals = 0u32;
        let s_hat = total_vol::<Cody>(beta, u, x, &mut evals).expect("round trip must converge");
        let beta_hat = normalised_otm_call::<Cody>(x, s_hat);
        let vol_ok = (s_hat - s).abs() <= 1e-10 * s;
        let price_ok = (beta_hat - beta).abs() <= 1e-9 * beta;
        assert!(
            vol_ok || price_ok,
            "x={x:e} s={s:e} s_hat={s_hat:e} beta={beta:e} beta_hat={beta_hat:e}"
        );
    }
}

/// Plain Cholesky: for n in 1..=6 and a symmetric matrix with positive
/// diagonal and raw correlations in [-1.5, 1.5], `cholesky_lower` returns
/// Ok or Err without panicking; on Ok, L is finite, lower triangular and
/// L L^T matches A to 1e-10 of the largest variance; on Err the row is < n.
pub fn check_cholesky_lower(data: &[u8]) {
    let mut fin = FuzzInput::new(data);
    let n = 1 + (fin.u64() % 6) as usize;
    let vol: Vec<f64> = (0..n).map(|_| fin.in_range(1e-4, 10.0)).collect();
    let mut a = vec![0.0; n * n];
    let mut vmax: f64 = 0.0;
    for i in 0..n {
        a[i * n + i] = vol[i] * vol[i];
        vmax = vmax.max(a[i * n + i]);
        for j in 0..i {
            let c = fin.in_range(-1.5, 1.5) * vol[i] * vol[j];
            a[i * n + j] = c;
            a[j * n + i] = c;
        }
    }
    let mut l = vec![0.0; n * n];
    match cholesky_lower(&a, n, &mut l) {
        Err(e) => assert!(e.row < n),
        Ok(()) => {
            for i in 0..n {
                for j in 0..n {
                    assert!(l[i * n + j].is_finite());
                    if j > i {
                        assert_eq!(l[i * n + j], 0.0);
                    }
                    let s: f64 = (0..=i.min(j)).map(|k| l[i * n + k] * l[j * n + k]).sum();
                    assert!(
                        (s - a[i * n + j]).abs() <= 1e-10 * vmax,
                        "({i},{j}) {s:e} vs {:e}",
                        a[i * n + j]
                    );
                }
            }
        }
    }
}
