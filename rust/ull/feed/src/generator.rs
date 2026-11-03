//! Seeded generator of Binance-shaped diff-depth days with a reference book.
//!
//! Each update id `n` changes one to three levels near a drifting mid. Events
//! group one or more consecutive ids (`U..=u`), as Binance does at 100 ms.
//! The reference book is the plain map applied in id order; the handler under
//! test must match it whenever it is not stale.

use std::collections::BTreeMap;

use crate::book::{SCALE, Side};

/// xorshift64* (Vigna 2016): small, seeded, reproducible.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Seeded generator (seed 0 is replaced by a constant).
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `0..n`.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Format fixed point (1e8) as a Binance-style 8-decimal string.
pub fn fmt_fixed(v: i64) -> String {
    format!("{}.{:08}", v / SCALE, v % SCALE)
}

/// One generated event: raw JSON plus its id range.
#[derive(Clone, Debug)]
pub struct Event {
    /// First id.
    pub first: u64,
    /// Last id.
    pub last: u64,
    /// Raw JSON as Binance would send it.
    pub raw: String,
}

/// Generator state and reference book.
#[derive(Clone, Debug)]
pub struct Generator {
    rng: Rng,
    next_id: u64,
    mid: i64,
    /// Reference bids (price -> qty).
    pub bids: BTreeMap<i64, i64>,
    /// Reference asks.
    pub asks: BTreeMap<i64, i64>,
}

const TICK: i64 = SCALE / 100; // 0.01

impl Generator {
    /// Start at id `start` with a seeded book around 100.00.
    pub fn new(seed: u64, start: u64) -> Self {
        let mut g = Self {
            rng: Rng::new(seed),
            next_id: start,
            mid: 100 * SCALE,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        };
        for i in 1..=20 {
            g.bids
                .insert(g.mid - i * TICK, (1 + g.rng.below(50) as i64) * SCALE / 10);
            g.asks
                .insert(g.mid + i * TICK, (1 + g.rng.below(50) as i64) * SCALE / 10);
        }
        g
    }

    /// Last id reflected in the reference book.
    pub fn last_id(&self) -> u64 {
        self.next_id - 1
    }

    /// REST snapshot of the reference book now, Binance shape.
    pub fn snapshot_json(&self) -> String {
        let lv = |m: &BTreeMap<i64, i64>, rev: bool| {
            let mut v: Vec<String> = m
                .iter()
                .map(|(p, q)| format!("[\"{}\",\"{}\"]", fmt_fixed(*p), fmt_fixed(*q)))
                .collect();
            if rev {
                v.reverse();
            }
            v.join(",")
        };
        format!(
            "{{\"lastUpdateId\":{},\"bids\":[{}],\"asks\":[{}]}}",
            self.last_id(),
            lv(&self.bids, true),
            lv(&self.asks, false)
        )
    }

    /// Next event covering 1-3 ids.
    pub fn next_event(&mut self) -> Event {
        let first = self.next_id;
        let n_ids = 1 + self.rng.below(3);
        let mut b = Vec::new();
        let mut a = Vec::new();
        for _ in 0..n_ids {
            if self.rng.below(20) == 0 {
                self.mid += (self.rng.below(3) as i64 - 1) * TICK;
            }
            for _ in 0..1 + self.rng.below(3) {
                let side = if self.rng.below(2) == 0 {
                    Side::Bid
                } else {
                    Side::Ask
                };
                let off = (1 + self.rng.below(25) as i64) * TICK;
                let qty = if self.rng.below(4) == 0 {
                    0
                } else {
                    (1 + self.rng.below(50) as i64) * SCALE / 10
                };
                let (px, book, list) = match side {
                    Side::Bid => (self.mid - off, &mut self.bids, &mut b),
                    Side::Ask => (self.mid + off, &mut self.asks, &mut a),
                };
                if qty == 0 {
                    book.remove(&px);
                } else {
                    book.insert(px, qty);
                }
                list.push(format!("[\"{}\",\"{}\"]", fmt_fixed(px), fmt_fixed(qty)));
            }
            self.next_id += 1;
        }
        let last = self.next_id - 1;
        let raw = format!(
            "{{\"e\":\"depthUpdate\",\"E\":{},\"s\":\"QPUSDT\",\"U\":{},\"u\":{},\"b\":[{}],\"a\":[{}]}}",
            1_700_000_000_000u64 + last,
            first,
            last,
            b.join(","),
            a.join(",")
        );
        Event { first, last, raw }
    }

    /// Reference levels of one side, best first (same shape as `CryptoBook::levels`).
    pub fn levels(&self, s: Side) -> Vec<(i64, i64)> {
        match s {
            Side::Bid => self.bids.iter().rev().map(|(&p, &q)| (p, q)).collect(),
            Side::Ask => self.asks.iter().map(|(&p, &q)| (p, q)).collect(),
        }
    }
}
