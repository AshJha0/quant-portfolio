//! Order statistics of bootstrap resamples drawn from the W4(b) Philox
//! streams ([`crate::path_rng`]), shared by the equity and FX VaR engines.
//!
//! The plain way to get the `m`-th largest loss of one resample is to copy
//! the `n` drawn losses into a buffer and run a selection, which costs a
//! few nanoseconds per draw. [`RankedLosses`] sorts the losses once per
//! call instead. A resample is then a count of how often each of the
//! largest few losses was drawn, and the `m`-th largest is found by walking
//! those counts in order. The answer is the same value, bit for bit: losses
//! are ordered by [`f64::total_cmp`], and two losses that compare equal
//! under it have the same bits.

use crate::path_rng::{LANES, philox_indices};

/// Integer key in the order of [`f64::total_cmp`] (the transform the
/// standard library uses): `a.total_cmp(&b) == total_key(a).cmp(&total_key(b))`.
#[inline]
pub fn total_key(v: f64) -> i64 {
    let b = v.to_bits() as i64;
    b ^ ((((b >> 63) as u64) >> 1) as i64)
}

/// Inverse of [`total_key`].
#[inline]
pub fn from_total_key(k: i64) -> f64 {
    f64::from_bits((k ^ ((((k >> 63) as u64) >> 1) as i64)) as u64)
}

/// The losses `-pnl[i]` of a P&L sample, ranked once, for repeated
/// bootstrap order statistics.
#[derive(Debug, Clone)]
pub struct RankedLosses {
    /// Losses, largest first (by `total_cmp`).
    sorted: Vec<f64>,
    /// `rank[i]`: position of `-pnl[i]` in `sorted` (ties broken by index).
    rank: Vec<u32>,
}

impl RankedLosses {
    /// Rank the losses of `pnl` (`pnl.len() < 2^32`).
    pub fn new(pnl: &[f64]) -> Self {
        let n = pnl.len();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            let (la, lb) = (-pnl[a as usize], -pnl[b as usize]);
            lb.total_cmp(&la).then(a.cmp(&b))
        });
        let mut rank = vec![0u32; n];
        for (r, &i) in order.iter().enumerate() {
            rank[i as usize] = r as u32;
        }
        let sorted = order.iter().map(|&i| -pnl[i as usize]).collect();
        RankedLosses { sorted, rank }
    }

    /// Number of losses.
    pub fn len(&self) -> usize {
        self.rank.len()
    }

    /// True when there are no losses.
    pub fn is_empty(&self) -> bool {
        self.rank.is_empty()
    }

    /// The `m`-th largest loss (`1 <= m <= n`) of bootstrap resample `b`:
    /// `n` indices `min(n - 1, floor(u n))` from Philox stream `(seed, b)`.
    /// The same bits as filling a buffer with the drawn losses and
    /// selecting the `m`-th largest by `total_cmp`. `counts` is scratch
    /// (resized here; reuse it across calls to avoid allocation).
    pub fn kth_largest(&self, m: usize, seed: u64, b: u64, counts: &mut Vec<u32>) -> f64 {
        let n = self.len();
        // Only ranks below `cut` are counted. A resample puts about `cut`
        // draws there (each rank is drawn once on average), far more than
        // `m`; when it does not, the walk below falls back to the full count.
        let cut = n.min(2 * m + 64);
        let mut v = self.walk(m, seed, b, cut, counts);
        if v.is_none() && cut < n {
            v = self.walk(m, seed, b, n, counts);
        }
        // With `cut == n` all n >= m draws are counted, so the walk always
        // finds the m-th; the fallback value is never used.
        v.unwrap_or(f64::NAN)
    }

    fn walk(&self, m: usize, seed: u64, b: u64, cut: usize, counts: &mut Vec<u32>) -> Option<f64> {
        let n = self.len();
        counts.clear();
        counts.resize(cut, 0);
        let mut idx = [0usize; 2 * LANES];
        let mut block = 0u64;
        let mut done = 0usize;
        while done < n {
            let len = (n - done).min(2 * LANES);
            philox_indices(seed, b, block, n, &mut idx);
            block += LANES as u64;
            for &i in &idx[..len] {
                let r = self.rank.get(i).map_or(usize::MAX, |&r| r as usize);
                if let Some(c) = counts.get_mut(r) {
                    *c += 1;
                }
            }
            done += len;
        }
        let mut cum = 0usize;
        for (r, &c) in counts.iter().enumerate() {
            cum += c as usize;
            if cum >= m {
                return self.sorted.get(r).copied();
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::path_rng::PathRng;

    fn by_selection(pnl: &[f64], m: usize, seed: u64, b: u64) -> f64 {
        let n = pnl.len();
        let mut rng = PathRng::new(seed, b);
        let mut buf: Vec<f64> = (0..n)
            .map(|_| -pnl[((rng.uniform() * n as f64) as usize).min(n - 1)])
            .collect();
        let (_, lm, _) = buf.select_nth_unstable_by(m - 1, |a, b| b.total_cmp(a));
        *lm
    }

    #[test]
    fn kth_largest_matches_selection_bit_for_bit() {
        let mut counts = Vec::new();
        let mut cases = 0;
        for n in [1usize, 2, 10, 31, 32, 33, 250, 1000, 4099] {
            // Ties and both signed zeros.
            let pnl: Vec<f64> = (0..n)
                .map(|i| match i % 7 {
                    0 => 0.0,
                    1 => -0.0,
                    2 => 5.0,
                    _ => ((i * 7919) % 1013) as f64 * 0.37 - 180.0,
                })
                .collect();
            let ranked = RankedLosses::new(&pnl);
            let ms = [1, 2, 3, n / 100, n / 40, n / 2, n];
            for &m in ms.iter().filter(|&&m| m >= 1 && m <= n) {
                for b in 0..6u64 {
                    let seed = 0xB007 ^ (m as u64) << 9 ^ n as u64;
                    let got = ranked.kth_largest(m, seed, b, &mut counts);
                    let want = by_selection(&pnl, m, seed, b);
                    assert_eq!(got.to_bits(), want.to_bits(), "n={n} m={m} b={b}");
                    cases += 1;
                }
            }
        }
        assert!(cases > 200);
    }

    #[test]
    fn total_key_orders_like_total_cmp() {
        let v = [
            f64::NEG_INFINITY,
            -1e300,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            1e300,
            f64::INFINITY,
        ];
        for &a in &v {
            assert_eq!(from_total_key(total_key(a)).to_bits(), a.to_bits());
            for &b in &v {
                assert_eq!(a.total_cmp(&b), total_key(a).cmp(&total_key(b)));
            }
        }
    }

    #[test]
    fn philox_indices_match_float_cast() {
        // The vectorised index path must equal `((u * n) as usize).min(n - 1)`
        // with `u` from the scalar stream, including n where `u n` lands
        // close to integers.
        for n in [
            1usize,
            2,
            3,
            7,
            250,
            1000,
            100_000,
            (1 << 31) + 3,
            (1 << 52) + 1,
        ] {
            for b in 0..4u64 {
                let mut rng = PathRng::new(77, b);
                let mut idx = [0usize; 2 * LANES];
                for block in (0..8u64).map(|k| k * LANES as u64) {
                    philox_indices(77, b, block, n, &mut idx);
                    for &i in &idx {
                        let want = ((rng.uniform() * n as f64) as usize).min(n - 1);
                        assert_eq!(i, want, "n={n} b={b}");
                    }
                }
            }
        }
    }
}
