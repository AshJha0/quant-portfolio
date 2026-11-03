//! Reference solutions for part B of the Rust pack.
//! Checked by `rustc --test solutions.rs` (see career-prep/tests/test_compiled_packs.py).

#![allow(dead_code)]

/// RB1: fixed capacity, power-of-two size so the index wrap is a mask.
pub struct Ring {
    buf: Box<[u64]>,
    head: usize, // next slot to read
    len: usize,
    mask: usize,
}

impl Ring {
    pub fn with_capacity_pow2(cap: usize) -> Self {
        assert!(cap.is_power_of_two(), "capacity must be a power of two");
        Ring { buf: vec![0; cap].into_boxed_slice(), head: 0, len: 0, mask: cap - 1 }
    }

    pub fn push(&mut self, x: u64) -> bool {
        if self.len == self.buf.len() {
            return false;
        }
        self.buf[(self.head + self.len) & self.mask] = x;
        self.len += 1;
        true
    }

    pub fn pop(&mut self) -> Option<u64> {
        if self.len == 0 {
            return None;
        }
        let x = self.buf[self.head];
        self.head = (self.head + 1) & self.mask;
        self.len -= 1;
        Some(x)
    }
}

/// RB2
pub fn vwap(trades: &[(f64, f64)]) -> Option<f64> {
    let (n, v) = trades.iter().fold((0.0, 0.0), |(n, v), &(p, q)| (n + p * q, v + q));
    if v == 0.0 {
        None
    } else {
        Some(n / v)
    }
}

/// RB3
pub fn max_drawdown(prices: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut worst: f64 = 0.0;
    for &p in prices {
        peak = peak.max(p);
        worst = worst.max((peak - p) / peak);
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    #[test]
    fn ring_matches_vecdeque() {
        let mut r = Ring::with_capacity_pow2(4);
        let mut q: VecDeque<u64> = VecDeque::new();
        let mut seed: u64 = 7;
        for i in 0..10_000u64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if seed >> 63 == 0 {
                let ok = r.push(i);
                assert_eq!(ok, q.len() < 4);
                if ok {
                    q.push_back(i);
                }
            } else {
                assert_eq!(r.pop(), q.pop_front());
            }
        }
    }

    #[test]
    fn vwap_basic() {
        assert_eq!(vwap(&[]), None);
        assert!((vwap(&[(10.0, 100.0), (11.0, 300.0)]).unwrap() - 10.75).abs() < 1e-12);
    }

    #[test]
    fn drawdown_matches_brute_force() {
        let xs = [100.0, 120.0, 90.0, 130.0, 65.0, 70.0];
        let mut brute: f64 = 0.0;
        for i in 0..xs.len() {
            for j in i..xs.len() {
                brute = brute.max((xs[i] - xs[j]) / xs[i]);
            }
        }
        assert!((max_drawdown(&xs) - brute).abs() < 1e-12);
        assert!((brute - 0.5).abs() < 1e-12);
    }
}
