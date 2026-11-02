//! O(n) order statistics and the type-7 quantile.
//!
//! `select_nth_unstable_by` (introselect, O(n) average) puts the k-th
//! smallest value exactly where a sort would, so the results here are the
//! same `f64` as the sort-based code: no tolerance applies. The type-7
//! quantile (Hyndman and Fan 1996, definition 7; NumPy's default) takes
//! `x(lo + 1)` as the minimum of the part above `lo`. Twin of `qp/quantile.hpp`.

/// The k-th smallest value (0-based) of `work`, which is reordered.
///
/// # Panics
/// If `k >= work.len()`.
#[must_use]
pub fn select_kth(work: &mut [f64], k: usize) -> f64 {
    *work.select_nth_unstable_by(k, f64::total_cmp).1
}

/// Type-7 quantile of `work` (reordered) at `q` in [0, 1]:
/// `h = (n - 1) q`, `x(lo) + (h - lo) (x(lo + 1) - x(lo))`.
///
/// # Panics
/// If `work` is empty.
#[must_use]
pub fn quantile_type7(work: &mut [f64], q: f64) -> f64 {
    let n = work.len();
    let h = (n - 1) as f64 * q;
    let lo = h.floor() as usize;
    let xlo = select_kth(work, lo);
    if lo + 1 >= n {
        return xlo;
    }
    let xhi = work[lo + 1..]
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(xlo);
    xlo + (h - lo as f64) * (xhi - xlo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn matches_sort_bit_for_bit() {
        for n in [1usize, 2, 3, 10, 101, 1000, 4097] {
            let mut g = SplitMix64::new(n as u64);
            let x: Vec<f64> = (0..n)
                .map(|_| (g.next_u64() >> 11) as f64 * 1e-12 - 4.5e3)
                .collect();
            let mut s = x.clone();
            s.sort_by(f64::total_cmp);
            for q in [0.0, 0.01, 0.025, 0.5, 0.975, 0.99, 1.0] {
                let h = (n - 1) as f64 * q;
                let lo = h.floor() as usize;
                let hi = (lo + 1).min(n - 1);
                let r = s[lo] + (h - lo as f64) * (s[hi] - s[lo]);
                assert_eq!(quantile_type7(&mut x.clone(), q).to_bits(), r.to_bits());
                assert_eq!(select_kth(&mut x.clone(), lo), s[lo]);
            }
        }
    }
}
