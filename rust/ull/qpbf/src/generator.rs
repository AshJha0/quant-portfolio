//! Seeded QPBF generator with a per-type ledger.
//!
//! Plain idea: keep a small model of each instrument's resting orders and
//! draw the next event from it, so every Modify, Cancel and Execute names an
//! order that is really live. The same seed always gives the same bytes, and
//! the ledger counts how many messages of each type were produced so that a
//! decoder can be checked against it.
//!
//! The plan has the W25 simulator drive this generator. W25 is a separate
//! lane; until it merges, the event mix below is a self-contained stand-in.
//! The seam is `Generator::next_msg`: a W25-driven source only needs to
//! produce the same `Msg` values.
//!
//! Randomness: xoshiro256** seeded through SplitMix64 (Blackman and Vigna,
//! "Scrambled linear pseudorandom number generators", 2021). Defined here,
//! not imported, so the C++ lane can reproduce identical streams if the two
//! sides agree to share generated days byte for byte.

use crate::{Msg, MsgType, N_TYPES, PacketWriter, SNAP_LAST, Side};
use std::collections::HashMap;

/// xoshiro256** seeded via SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng([u64; 4]);

impl Rng {
    /// Seeds the four state words from SplitMix64(seed).
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut s = [0u64; 4];
        for w in &mut s {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *w = x ^ (x >> 31);
        }
        Rng(s)
    }
    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }
    /// Uniform in `[0, n)` by Lemire's multiply-shift (bias below 2^-32 for
    /// the small `n` used here; acceptable for a test-data generator).
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }
}

/// Generator settings.
#[derive(Clone, Debug)]
pub struct GenConfig {
    /// Seed.
    pub seed: u64,
    /// Number of instruments, ids `1..=instruments`.
    pub instruments: u32,
    /// Starting mid price in ticks.
    pub start_mid: i64,
    /// Orders rest within this many ticks of the mid.
    pub band: i64,
    /// Event weights in ledger order: Add, Modify, Cancel, Execute, Trade,
    /// (Snapshot unused in the random stream), Heartbeat.
    pub weights: [u32; N_TYPES],
    /// When an instrument has this many live orders an Add becomes a Cancel,
    /// so the book size stays bounded on long days.
    pub max_live: usize,
}

impl Default for GenConfig {
    fn default() -> Self {
        GenConfig {
            seed: 1,
            instruments: 4,
            start_mid: 10_000,
            band: 50,
            weights: [40, 15, 25, 12, 5, 0, 3],
            max_live: 2_000,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Live {
    id: u64,
    side: Side,
    price: i64,
    qty: u32,
    /// Arrival rank; time priority within a level.
    rank: u64,
}

#[derive(Debug, Default)]
struct Inst {
    mid: i64,
    orders: Vec<Live>,
    index: HashMap<u64, usize>,
}

/// Deterministic event source.
#[derive(Debug)]
pub struct Generator {
    cfg: GenConfig,
    rng: Rng,
    insts: Vec<Inst>,
    next_order: u64,
    next_trade: u64,
    rank: u64,
    ts: u64,
    ledger: [u64; N_TYPES],
}

impl Generator {
    /// New generator.
    pub fn new(cfg: GenConfig) -> Self {
        assert!(cfg.instruments >= 1 && cfg.band >= 1 && cfg.start_mid > cfg.band);
        let insts = (0..cfg.instruments)
            .map(|_| Inst {
                mid: cfg.start_mid,
                ..Inst::default()
            })
            .collect();
        Generator {
            rng: Rng::new(cfg.seed),
            cfg,
            insts,
            next_order: 1,
            next_trade: 1,
            rank: 0,
            ts: 0,
            ledger: [0; N_TYPES],
        }
    }

    /// Messages produced so far, by type (ledger order of `MsgType::ALL`).
    pub fn ledger(&self) -> [u64; N_TYPES] {
        self.ledger
    }

    /// Live order count on `instrument`.
    pub fn live_orders(&self, instrument: u32) -> usize {
        self.insts[(instrument - 1) as usize].orders.len()
    }

    fn pick_type(&mut self) -> MsgType {
        let total: u32 = self.cfg.weights.iter().sum();
        let mut r = self.rng.below(total as u64) as u32;
        for (i, w) in self.cfg.weights.iter().enumerate() {
            if r < *w {
                return MsgType::ALL[i];
            }
            r -= w;
        }
        MsgType::Add
    }

    fn remove(inst: &mut Inst, pos: usize) {
        let gone = inst.orders.swap_remove(pos);
        inst.index.remove(&gone.id);
        if pos < inst.orders.len() {
            let moved = inst.orders[pos].id;
            inst.index.insert(moved, pos);
        }
    }

    /// Draws the next message and updates the model and the ledger.
    pub fn next_msg(&mut self) -> Msg {
        self.ts += 1 + self.rng.below(1000);
        let ts_ns = self.ts;
        let ii = self.rng.below(self.cfg.instruments as u64) as usize;
        let instrument = ii as u32 + 1;
        let mut kind = self.pick_type();
        if self.insts[ii].orders.is_empty()
            && matches!(kind, MsgType::Modify | MsgType::Cancel | MsgType::Execute)
        {
            kind = MsgType::Add;
        }
        if kind == MsgType::Add && self.insts[ii].orders.len() >= self.cfg.max_live {
            kind = MsgType::Cancel;
        }
        let band = self.cfg.band;
        // Mid drifts by at most one tick per event, kept well above zero.
        let step = self.rng.below(3) as i64 - 1;
        let floor = band + 1;
        let inst = &mut self.insts[ii];
        inst.mid = (inst.mid + step).max(floor);
        let m = match kind {
            MsgType::Add => {
                let side = if self.rng.below(2) == 0 {
                    Side::Buy
                } else {
                    Side::Sell
                };
                let off = 1 + self.rng.below(band as u64) as i64;
                let price = match side {
                    Side::Buy => inst.mid - off,
                    Side::Sell => inst.mid + off,
                };
                let qty = 1 + self.rng.below(500) as u32;
                let id = self.next_order;
                self.next_order += 1;
                self.rank += 1;
                inst.index.insert(id, inst.orders.len());
                inst.orders.push(Live {
                    id,
                    side,
                    price,
                    qty,
                    rank: self.rank,
                });
                Msg::Add {
                    instrument,
                    ts_ns,
                    order_id: id,
                    side,
                    price,
                    qty,
                }
            }
            MsgType::Modify => {
                let pos = self.rng.below(inst.orders.len() as u64) as usize;
                let o = inst.orders[pos];
                let new_price = if self.rng.below(2) == 0 {
                    o.price
                } else {
                    let off = 1 + self.rng.below(band as u64) as i64;
                    match o.side {
                        Side::Buy => inst.mid - off,
                        Side::Sell => inst.mid + off,
                    }
                };
                let new_qty = 1 + self.rng.below(500) as u32;
                let keeps = new_price == o.price && new_qty <= o.qty;
                let l = &mut inst.orders[pos];
                l.price = new_price;
                l.qty = new_qty;
                if !keeps {
                    self.rank += 1;
                    l.rank = self.rank;
                }
                Msg::Modify {
                    instrument,
                    ts_ns,
                    order_id: o.id,
                    price: new_price,
                    qty: new_qty,
                }
            }
            MsgType::Cancel => {
                // Half of cancels remove the whole order, half a random part
                // (spec section 5 rule 3: Cancel reduces by qty).
                let pos = self.rng.below(inst.orders.len() as u64) as usize;
                let o = inst.orders[pos];
                let qty = if self.rng.below(2) == 0 {
                    o.qty
                } else {
                    1 + self.rng.below(o.qty as u64) as u32
                };
                if qty == o.qty {
                    Self::remove(inst, pos);
                } else {
                    inst.orders[pos].qty -= qty;
                }
                Msg::Cancel {
                    instrument,
                    ts_ns,
                    order_id: o.id,
                    qty,
                }
            }
            MsgType::Execute => {
                let pos = self.rng.below(inst.orders.len() as u64) as usize;
                let o = inst.orders[pos];
                let qty = 1 + self.rng.below(o.qty as u64) as u32;
                if qty == o.qty {
                    Self::remove(inst, pos);
                } else {
                    inst.orders[pos].qty -= qty;
                }
                let tid = self.next_trade;
                self.next_trade += 1;
                Msg::Execute {
                    instrument,
                    ts_ns,
                    order_id: o.id,
                    match_id: tid,
                    qty,
                }
            }
            MsgType::Trade => {
                let side = if self.rng.below(2) == 0 {
                    Side::Buy
                } else {
                    Side::Sell
                };
                let tid = self.next_trade;
                self.next_trade += 1;
                Msg::Trade {
                    instrument,
                    ts_ns,
                    side,
                    match_id: tid,
                    price: inst.mid,
                    qty: 1 + self.rng.below(200) as u32,
                }
            }
            MsgType::Heartbeat | MsgType::Snapshot => Msg::Heartbeat {
                instrument: 0,
                ts_ns,
            },
        };
        self.ledger[m.kind() as usize] += 1;
        m
    }

    /// Snapshot of `instrument` as of feed sequence `as_of_seq` (spec
    /// section 6): one message per resting order, bids then asks, best price
    /// first within a side, queue order within a level. The last message has
    /// `SNAP_LAST`; an empty book is the end marker alone. Snapshot messages
    /// are counted in the ledger.
    pub fn snapshot(&mut self, instrument: u32, as_of_seq: u64) -> Vec<Msg> {
        let inst = &self.insts[(instrument - 1) as usize];
        let mut v: Vec<Live> = inst.orders.clone();
        v.sort_by_key(|o| match o.side {
            Side::Buy => (0, -o.price, o.rank),
            Side::Sell => (1, o.price, o.rank),
        });
        let ts_ns = self.ts;
        let out: Vec<Msg> = if v.is_empty() {
            vec![Msg::snapshot_end(instrument, ts_ns, as_of_seq)]
        } else {
            let n = v.len();
            v.iter()
                .enumerate()
                .map(|(i, o)| Msg::Snapshot {
                    instrument,
                    ts_ns,
                    flags: if i + 1 == n { SNAP_LAST } else { 0 },
                    order_id: o.id,
                    side: o.side,
                    price: o.price,
                    qty: o.qty,
                    as_of_seq,
                })
                .collect()
        };
        self.ledger[MsgType::Snapshot as usize] += out.len() as u64;
        out
    }
}

/// Generates `n_msgs` messages, packs them into packets (at most 1,472
/// bytes and 255 messages) starting at sequence 1, and hands each packet to
/// `sink`. Returns the ledger.
pub fn generate_day(
    cfg: GenConfig,
    session_id: u32,
    n_msgs: u64,
    mut sink: impl FnMut(&[u8]),
) -> [u64; N_TYPES] {
    let mut g = Generator::new(cfg);
    let mut w = PacketWriter::new(session_id, 1);
    for _ in 0..n_msgs {
        let m = g.next_msg();
        if w.would_overflow(&m) {
            w.flush(&mut sink);
        }
        w.push(&m);
    }
    if !w.is_empty() {
        w.flush(&mut sink);
    }
    g.ledger()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_seed_matches_reference_vector() {
        // SplitMix64 from seed 0: first output 0xE220A8397B1DCDAF
        // (Vigna's reference splitmix64.c).
        let r = Rng::new(0);
        assert_eq!(r.0[0], 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn same_seed_same_bytes() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let la = generate_day(GenConfig::default(), 1, 5_000, |p| a.extend_from_slice(p));
        let lb = generate_day(GenConfig::default(), 1, 5_000, |p| b.extend_from_slice(p));
        assert_eq!(a, b);
        assert_eq!(la, lb);
        assert_eq!(la.iter().sum::<u64>(), 5_000);
    }
}
