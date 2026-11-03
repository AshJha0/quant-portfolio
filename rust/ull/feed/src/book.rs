//! Normalised feed events and the crypto book they update.
//!
//! Plain idea: Coinbase and Binance send prices and sizes as decimal strings.
//! They are parsed once into fixed-point integers (`SCALE` = 1e8, so eight
//! decimal places, which covers both venues' tick and lot sizes) and never
//! touch a float. A level feed (Binance diff-depth, Coinbase level2) says
//! "the total at price p is now q"; an order feed (Coinbase full) says what
//! happened to one order, and the level totals are the sum over its orders.
//!
//! Formal statement: after applying a snapshot with last id `L` and then every
//! delta with ids `L+1..=n` in order, `levels(side)` equals the venue's book at
//! id `n`. A level update with `q = 0` removes the level. For the order feed,
//! `levels` is kept equal to the per-price sum of live order sizes after every
//! change (checked by `check_invariant` in the tests).
//!
//! Worked example: snapshot bids {100.00: 2}, then delta bid 100.00 -> 0 and
//! bid 99.50 -> 1.5 gives bids {99.50: 1.5}; `best_bid()` is 99.50.
//!
//! Seam: the part-1 crate `qp-ull-book` holds the allocation-free L3 book on
//! integer ticks and u64 ids. Crypto ids are UUIDs and quantities are decimals,
//! so this book uses ordered maps. Swapping it for the flat book needs an
//! id-interning table and a tick window per product; that is listed as an
//! integration item, not done here.

use std::collections::{BTreeMap, HashMap};

/// Fixed-point scale: 1e8.
pub const SCALE: i64 = 100_000_000;

/// Book side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Side {
    /// Bids (buy orders).
    Bid,
    /// Asks (sell orders).
    Ask,
}

/// Parse a non-negative decimal string into fixed point (1e8).
///
/// Rejects signs, exponents, empty parts and more than eight significant
/// decimals (trailing zeros past the eighth place are allowed, since Binance
/// pads to its precision).
pub fn parse_fixed(s: &str) -> Option<i64> {
    let (ip, fp) = match s.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    if ip.is_empty()
        || !ip.bytes().all(|c| c.is_ascii_digit())
        || !fp.bytes().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    if s.contains('.') && fp.is_empty() {
        return None;
    }
    let fp_trim = fp.trim_end_matches('0');
    if fp_trim.len() > 8 {
        return None;
    }
    let i: i64 = ip.parse().ok()?;
    let mut f: i64 = if fp_trim.is_empty() {
        0
    } else {
        fp_trim.parse().ok()?
    };
    for _ in fp_trim.len()..8 {
        f *= 10;
    }
    i.checked_mul(SCALE)?.checked_add(f)
}

/// One change carried by a delta.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Change {
    /// Level feed: total at `px` on `side` is now `qty` (0 removes).
    Level {
        /// Side.
        side: Side,
        /// Price (1e8).
        px: i64,
        /// New total (1e8).
        qty: i64,
    },
    /// Order feed: an order rests on the book.
    Open {
        /// Order id (UUID as u128).
        id: u128,
        /// Side.
        side: Side,
        /// Price (1e8).
        px: i64,
        /// Remaining size (1e8).
        qty: i64,
    },
    /// Order feed: the order left the book (filled or cancelled).
    Done {
        /// Order id.
        id: u128,
    },
    /// Order feed: the resting order traded `qty`.
    Match {
        /// Maker order id.
        id: u128,
        /// Traded size (1e8).
        qty: i64,
    },
    /// Order feed: the order's size changed to `qty`.
    Resize {
        /// Order id.
        id: u128,
        /// New size (1e8).
        qty: i64,
    },
}

/// A sequenced delta covering update ids `first..=last`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Delta {
    /// First update id covered.
    pub first: u64,
    /// Last update id covered.
    pub last: u64,
    /// Changes in venue order. May be empty (Coinbase `received`).
    pub changes: Vec<Change>,
}

/// A book snapshot as of update id `last`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// Last update id reflected.
    pub last: u64,
    /// Level entries (level feeds).
    pub levels: Vec<(Side, i64, i64)>,
    /// Order entries (order feeds): id, side, px, qty.
    pub orders: Vec<(u128, Side, i64, i64)>,
}

/// Why a change could not be applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// Order feed referenced an unknown order id.
    UnknownOrder,
    /// Match larger than the resting size, or a negative size.
    BadSize,
}

/// L2 levels plus (for order feeds) the live orders behind them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CryptoBook {
    bids: BTreeMap<i64, i64>,
    asks: BTreeMap<i64, i64>,
    orders: HashMap<u128, (Side, i64, i64)>,
}

impl CryptoBook {
    /// Empty book.
    pub fn new() -> Self {
        Self::default()
    }

    fn side_mut(&mut self, s: Side) -> &mut BTreeMap<i64, i64> {
        match s {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        }
    }

    fn add_level(&mut self, s: Side, px: i64, dq: i64) {
        let m = self.side_mut(s);
        let e = m.entry(px).or_insert(0);
        *e += dq;
        if *e == 0 {
            m.remove(&px);
        }
    }

    /// Replace the whole book with a snapshot.
    pub fn reset(&mut self, snap: &Snapshot) {
        self.bids.clear();
        self.asks.clear();
        self.orders.clear();
        for &(s, px, q) in &snap.levels {
            if q > 0 {
                self.side_mut(s).insert(px, q);
            }
        }
        for &(id, s, px, q) in &snap.orders {
            self.orders.insert(id, (s, px, q));
            self.add_level(s, px, q);
        }
    }

    /// Apply one change.
    pub fn apply(&mut self, c: &Change) -> Result<(), ApplyError> {
        match *c {
            Change::Level { side, px, qty } => {
                if qty < 0 {
                    return Err(ApplyError::BadSize);
                }
                if qty == 0 {
                    self.side_mut(side).remove(&px);
                } else {
                    self.side_mut(side).insert(px, qty);
                }
            }
            Change::Open { id, side, px, qty } => {
                if qty <= 0 {
                    return Err(ApplyError::BadSize);
                }
                if let Some((s0, p0, q0)) = self.orders.insert(id, (side, px, qty)) {
                    self.add_level(s0, p0, -q0);
                }
                self.add_level(side, px, qty);
            }
            Change::Done { id } => {
                // Coinbase sends `done` for orders that never rested (fully
                // filled on arrival); those are not on the book and are ignored.
                if let Some((s, p, q)) = self.orders.remove(&id) {
                    self.add_level(s, p, -q);
                }
            }
            Change::Match { id, qty } => {
                let (s, p, q) = *self.orders.get(&id).ok_or(ApplyError::UnknownOrder)?;
                if qty <= 0 || qty > q {
                    return Err(ApplyError::BadSize);
                }
                self.add_level(s, p, -qty);
                if qty == q {
                    self.orders.remove(&id);
                } else {
                    self.orders.insert(id, (s, p, q - qty));
                }
            }
            Change::Resize { id, qty } => {
                if qty < 0 {
                    return Err(ApplyError::BadSize);
                }
                // A change for an order not on the book is a received-only order.
                if let Some(&(s, p, q)) = self.orders.get(&id) {
                    self.add_level(s, p, qty - q);
                    if qty == 0 {
                        self.orders.remove(&id);
                    } else {
                        self.orders.insert(id, (s, p, qty));
                    }
                }
            }
        }
        Ok(())
    }

    /// Best bid (price, total).
    pub fn best_bid(&self) -> Option<(i64, i64)> {
        self.bids.iter().next_back().map(|(&p, &q)| (p, q))
    }
    /// Best ask (price, total).
    pub fn best_ask(&self) -> Option<(i64, i64)> {
        self.asks.iter().next().map(|(&p, &q)| (p, q))
    }
    /// Levels of one side, best first.
    pub fn levels(&self, s: Side) -> Vec<(i64, i64)> {
        match s {
            Side::Bid => self.bids.iter().rev().map(|(&p, &q)| (p, q)).collect(),
            Side::Ask => self.asks.iter().map(|(&p, &q)| (p, q)).collect(),
        }
    }
    /// Number of live orders (order feeds).
    pub fn order_count(&self) -> usize {
        self.orders.len()
    }
    /// Levels equal the per-price sum of live orders (order feeds only).
    pub fn check_invariant(&self) -> bool {
        if self.orders.is_empty() {
            return true;
        }
        let mut b = BTreeMap::new();
        let mut a = BTreeMap::new();
        for &(s, p, q) in self.orders.values() {
            *match s {
                Side::Bid => &mut b,
                Side::Ask => &mut a,
            }
            .entry(p)
            .or_insert(0) += q;
        }
        b == self.bids && a == self.asks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_point() {
        assert_eq!(parse_fixed("0.00012300"), Some(12_300));
        assert_eq!(parse_fixed("100"), Some(100 * SCALE));
        assert_eq!(parse_fixed("99.5"), Some(9_950_000_000));
        assert_eq!(parse_fixed("1.0000000000"), Some(SCALE));
        for bad in ["", ".5", "5.", "-1", "1e3", "0.000000001", "1.2.3", " 1"] {
            assert_eq!(parse_fixed(bad), None, "{bad}");
        }
    }

    #[test]
    fn worked_example() {
        let mut b = CryptoBook::new();
        b.reset(&Snapshot {
            last: 1,
            levels: vec![(Side::Bid, 100 * SCALE, 2 * SCALE)],
            orders: vec![],
        });
        b.apply(&Change::Level {
            side: Side::Bid,
            px: 100 * SCALE,
            qty: 0,
        })
        .unwrap();
        b.apply(&Change::Level {
            side: Side::Bid,
            px: 9_950_000_000,
            qty: 150_000_000,
        })
        .unwrap();
        assert_eq!(b.best_bid(), Some((9_950_000_000, 150_000_000)));
        assert_eq!(b.best_ask(), None);
    }

    #[test]
    fn order_feed_keeps_levels_equal_to_orders() {
        let mut b = CryptoBook::new();
        b.apply(&Change::Open {
            id: 1,
            side: Side::Ask,
            px: 10,
            qty: 5,
        })
        .unwrap();
        b.apply(&Change::Open {
            id: 2,
            side: Side::Ask,
            px: 10,
            qty: 3,
        })
        .unwrap();
        b.apply(&Change::Match { id: 1, qty: 2 }).unwrap();
        assert_eq!(b.best_ask(), Some((10, 6)));
        b.apply(&Change::Resize { id: 2, qty: 1 }).unwrap();
        b.apply(&Change::Done { id: 1 }).unwrap();
        b.apply(&Change::Done { id: 99 }).unwrap(); // never rested: ignored
        assert_eq!(b.best_ask(), Some((10, 1)));
        assert!(b.check_invariant());
        assert_eq!(
            b.apply(&Change::Match { id: 2, qty: 5 }),
            Err(ApplyError::BadSize)
        );
        assert_eq!(
            b.apply(&Change::Match { id: 7, qty: 1 }),
            Err(ApplyError::UnknownOrder)
        );
    }
}
