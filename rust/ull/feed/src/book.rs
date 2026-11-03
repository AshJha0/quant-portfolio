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
//! Storage: the book is the part-1 allocation-free flat-array book
//! (`qp_ull_book::Book`, integer ticks, u64 ids, u32 sizes). A per-product
//! [`ProductSpec`] maps decimal prices to ticks inside a fixed window and
//! decimal sizes to whole lots; Coinbase UUIDs are interned to u64 handles in
//! a fixed-capacity table. A price outside the window (or off the tick grid)
//! is [`ApplyError::OutOfWindow`]: the sync engine marks the book stale and
//! recovers from a snapshot, which re-centres the window. After construction,
//! applying a change never allocates (tests/no_alloc.rs).

use std::collections::BTreeMap;

use qp_ull_book::{Book, BookError};
use qp_ull_qpbf::Side as QSide;

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

/// Per-product layout of the flat book.
///
/// Prices are stored as integer ticks (`px / tick`) inside a window of
/// `n_levels` ticks; sizes as integer lots (`qty / lot`, at most `u32::MAX`
/// lots per order or per level-feed level). Both `tick` and `lot` are in
/// fixed-point units (1e8). The window is centred on the snapshot's touch
/// every time a snapshot is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductSpec {
    /// Price increment, fixed point (e.g. 0.01 = `SCALE / 100`).
    pub tick: i64,
    /// Size increment, fixed point (1 = 1e-8).
    pub lot: i64,
    /// Window width in ticks.
    pub n_levels: u32,
    /// Live orders (order feeds) or levels (level feeds) the book can hold.
    pub max_orders: u32,
}

impl Default for ProductSpec {
    /// 0.01 tick, 1e-8 lot (so up to ~42.9 units per order), a window of
    /// 65,536 ticks (+-327.68 around the touch at a 0.01 tick) and 65,536
    /// orders. Size it per product for real use.
    fn default() -> Self {
        Self {
            tick: SCALE / 100,
            lot: 1,
            n_levels: 1 << 16,
            max_orders: 1 << 16,
        }
    }
}

/// Fixed-capacity UUID -> u64 handle table (id interning). Linear probing
/// with backward-shift delete (Knuth, TAOCP vol. 3, 6.4, Algorithm R), and a
/// pre-allocated stack of free handles. Nothing allocates after `new`.
struct Interner {
    keys: Box<[u128]>,
    vals: Box<[u64]>, // 0 = empty slot
    mask: usize,
    free: Box<[u64]>,
    top: usize,
}

/// Handles carry the top bit so they never meet level-feed ids.
const HANDLE_BIT: u64 = 1 << 63;

impl Interner {
    fn new(cap: u32) -> Self {
        let n = (cap as usize * 2).max(2).next_power_of_two();
        Self {
            keys: vec![0; n].into_boxed_slice(),
            vals: vec![0; n].into_boxed_slice(),
            mask: n - 1,
            free: (0..cap as u64)
                .rev()
                .map(|i| HANDLE_BIT | (i + 1))
                .collect(),
            top: cap as usize,
        }
    }
    #[inline]
    fn home(&self, k: u128) -> usize {
        let h = (k as u64) ^ ((k >> 64) as u64);
        (h.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & self.mask
    }
    #[inline]
    fn get(&self, k: u128) -> Option<u64> {
        let mut i = self.home(k);
        loop {
            let v = self.vals[i];
            if v == 0 {
                return None;
            }
            if self.keys[i] == k {
                return Some(v);
            }
            i = (i + 1) & self.mask;
        }
    }
    /// Interns `k` (must not be present). `None` when every handle is in use.
    #[inline]
    fn insert(&mut self, k: u128) -> Option<u64> {
        if self.top == 0 {
            return None;
        }
        self.top -= 1;
        let h = self.free[self.top];
        let mut i = self.home(k);
        while self.vals[i] != 0 {
            i = (i + 1) & self.mask;
        }
        self.keys[i] = k;
        self.vals[i] = h;
        Some(h)
    }
    #[inline]
    fn remove(&mut self, k: u128) {
        let mut i = self.home(k);
        loop {
            if self.vals[i] == 0 {
                return;
            }
            if self.keys[i] == k {
                break;
            }
            i = (i + 1) & self.mask;
        }
        self.free[self.top] = self.vals[i];
        self.top += 1;
        let mut hole = i;
        let mut j = i;
        loop {
            j = (j + 1) & self.mask;
            if self.vals[j] == 0 {
                break;
            }
            let h = self.home(self.keys[j]);
            // Move j into the hole unless its home lies cyclically in (hole, j].
            let in_range = if hole <= j {
                hole < h && h <= j
            } else {
                hole < h || h <= j
            };
            if !in_range {
                self.keys[hole] = self.keys[j];
                self.vals[hole] = self.vals[j];
                hole = j;
            }
        }
        self.vals[hole] = 0;
    }
    fn clear(&mut self) {
        self.vals.fill(0);
        let cap = self.free.len();
        for (i, h) in self.free.iter_mut().enumerate() {
            *h = HANDLE_BIT | (cap - i) as u64;
        }
        self.top = cap;
    }
    fn len(&self) -> usize {
        self.free.len() - self.top
    }
    fn iter(&self) -> impl Iterator<Item = (u128, u64)> + '_ {
        self.keys
            .iter()
            .zip(self.vals.iter())
            .filter(|(_, v)| **v != 0)
            .map(|(k, v)| (*k, *v))
    }
}

/// Why a change could not be applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// Order feed referenced an unknown order id.
    UnknownOrder,
    /// Match larger than the resting size, or a negative size.
    BadSize,
    /// Price outside the product's window, or not a multiple of the tick:
    /// the engine marks the book stale and recovers from a snapshot, which
    /// re-centres the window.
    OutOfWindow,
    /// Size not a whole number of lots, or more than `u32::MAX` lots.
    Unrepresentable,
    /// Every order slot (or interned id) is in use.
    Full,
}

/// L2 levels plus (for order feeds) the live orders behind them, held in the
/// part-1 flat-array book (`qp_ull_book::Book`).
///
/// A level feed stores each level as one synthetic order whose id is
/// `2 * tick + side + 1`; an order feed interns each UUID to a u64 handle.
/// After construction no operation on this book allocates, except
/// [`CryptoBook::reset`] when the snapshot's touch has left the middle half
/// of the window (it then rebuilds the book around the new touch).
pub struct CryptoBook {
    spec: ProductSpec,
    book: Book,
    base: i64, // first tick of the window
    ids: Interner,
    /// Snapshot entries dropped because they lay outside the window.
    pub clipped: u64,
}

fn qside(s: Side) -> QSide {
    match s {
        Side::Bid => QSide::Buy,
        Side::Ask => QSide::Sell,
    }
}

impl Default for CryptoBook {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CryptoBook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CryptoBook")
            .field("spec", &self.spec)
            .field("bids", &self.levels(Side::Bid))
            .field("asks", &self.levels(Side::Ask))
            .field("orders", &self.order_count())
            .finish()
    }
}

/// Two books are equal when their levels and live orders are equal (the
/// window position and internal handles do not matter).
impl PartialEq for CryptoBook {
    fn eq(&self, o: &Self) -> bool {
        self.levels(Side::Bid) == o.levels(Side::Bid)
            && self.levels(Side::Ask) == o.levels(Side::Ask)
            && self.orders() == o.orders()
    }
}

impl Eq for CryptoBook {}

impl CryptoBook {
    /// Empty book with the default [`ProductSpec`].
    pub fn new() -> Self {
        Self::with_spec(ProductSpec::default())
    }

    /// Empty book for one product. All memory is allocated here.
    pub fn with_spec(spec: ProductSpec) -> Self {
        assert!(spec.tick > 0 && spec.lot > 0 && spec.n_levels >= 4 && spec.max_orders > 0);
        Self {
            spec,
            book: Book::new(0, spec.n_levels, spec.max_orders),
            base: 0,
            ids: Interner::new(spec.max_orders),
            clipped: 0,
        }
    }

    /// The price window `[lo, hi)` in fixed point.
    pub fn window(&self) -> (i64, i64) {
        let t = self.spec.tick;
        (self.base * t, (self.base + self.spec.n_levels as i64) * t)
    }

    /// The product layout.
    pub fn spec(&self) -> ProductSpec {
        self.spec
    }

    #[inline]
    fn tick_of(&self, px: i64) -> Result<i64, ApplyError> {
        if px < 0 || px % self.spec.tick != 0 {
            return Err(ApplyError::OutOfWindow);
        }
        Ok(px / self.spec.tick)
    }

    #[inline]
    fn lots_of(&self, qty: i64) -> Result<u32, ApplyError> {
        if qty % self.spec.lot != 0 {
            return Err(ApplyError::Unrepresentable);
        }
        u32::try_from(qty / self.spec.lot).map_err(|_| ApplyError::Unrepresentable)
    }

    #[inline]
    fn level_id(t: i64, s: Side) -> u64 {
        (t as u64) * 2 + (s == Side::Ask) as u64 + 1
    }

    #[inline]
    fn map(e: BookError) -> ApplyError {
        match e {
            BookError::OutOfRange(_) => ApplyError::OutOfWindow,
            BookError::Full => ApplyError::Full,
            BookError::UnknownOrder(_) => ApplyError::UnknownOrder,
            BookError::DuplicateOrder(_) | BookError::BadQty => ApplyError::BadSize,
        }
    }

    /// Replace the whole book with a snapshot. The window is re-centred on
    /// the snapshot's touch when the touch has left the middle half of the
    /// current window. Entries outside the window are dropped and counted in
    /// `clipped` (SPEC CHOICE: the book is exact inside its window; size the
    /// window to cover the snapshot depth you request).
    pub fn reset(&mut self, snap: &Snapshot) -> Result<(), ApplyError> {
        let mut bb: Option<i64> = None;
        let mut ba: Option<i64> = None;
        let entries = snap
            .levels
            .iter()
            .map(|&(s, p, q)| (s, p, q))
            .chain(snap.orders.iter().map(|&(_, s, p, q)| (s, p, q)));
        for (s, p, q) in entries {
            if q <= 0 {
                continue;
            }
            let t = self.tick_of(p)?;
            match s {
                Side::Bid => bb = Some(bb.map_or(t, |b| b.max(t))),
                Side::Ask => ba = Some(ba.map_or(t, |a| a.min(t))),
            }
        }
        let n = self.spec.n_levels as i64;
        let centre = match (bb, ba) {
            (Some(b), Some(a)) => Some((b + a) / 2),
            (Some(b), None) => Some(b),
            (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        let base = self.base;
        match centre {
            Some(c) if c < base + n / 4 || c >= base + 3 * n / 4 => {
                // Rebuild around the new touch: an allocation, on the
                // snapshot path only.
                self.base = (c - n / 2).max(0);
                self.book = Book::new(self.base, self.spec.n_levels, self.spec.max_orders);
            }
            _ => self.book.clear(),
        }
        self.ids.clear();
        for &(s, p, q) in &snap.levels {
            if q > 0 {
                let t = self.tick_of(p)?;
                let lots = self.lots_of(q)?;
                match self.book.add(Self::level_id(t, s), qside(s), t, lots) {
                    Err(BookError::OutOfRange(_)) => self.clipped += 1,
                    r => r.map_err(Self::map)?,
                }
            }
        }
        for &(id, s, p, q) in &snap.orders {
            self.open(id, s, p, q, true)?;
        }
        Ok(())
    }

    fn open(&mut self, id: u128, s: Side, px: i64, qty: i64, clip: bool) -> Result<(), ApplyError> {
        if qty <= 0 {
            return Err(ApplyError::BadSize);
        }
        let t = self.tick_of(px)?;
        let lots = self.lots_of(qty)?;
        if let Some(h) = self.ids.get(id) {
            // Re-open of a live id replaces it.
            let (_, _, q0) = self.book.order(h).ok_or(ApplyError::UnknownOrder)?;
            self.book.cancel(h, q0).map_err(Self::map)?;
            self.ids.remove(id);
        }
        let h = self.ids.insert(id).ok_or(ApplyError::Full)?;
        match self.book.add(h, qside(s), t, lots) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.ids.remove(id);
                if clip && matches!(e, BookError::OutOfRange(_)) {
                    self.clipped += 1;
                    Ok(())
                } else {
                    Err(Self::map(e))
                }
            }
        }
    }

    /// Apply one change. Does not allocate.
    #[inline]
    pub fn apply(&mut self, c: &Change) -> Result<(), ApplyError> {
        match *c {
            Change::Level { side, px, qty } => {
                if qty < 0 {
                    return Err(ApplyError::BadSize);
                }
                let t = self.tick_of(px)?;
                let id = Self::level_id(t, side);
                if qty == 0 {
                    // Removing an absent level is a no-op, as before; that
                    // includes a level outside the window, which this book
                    // never holds.
                    if let Some((_, _, q0)) = self.book.order(id) {
                        self.book.cancel(id, q0).map_err(Self::map)?;
                    }
                    return Ok(());
                }
                let lots = self.lots_of(qty)?;
                if self.book.order(id).is_some() {
                    self.book.modify(id, t, lots).map_err(Self::map)
                } else {
                    self.book.add(id, qside(side), t, lots).map_err(Self::map)
                }
            }
            Change::Open { id, side, px, qty } => self.open(id, side, px, qty, false),
            Change::Done { id } => {
                // Coinbase sends `done` for orders that never rested (fully
                // filled on arrival); those are not on the book and are ignored.
                if let Some(h) = self.ids.get(id) {
                    let (_, _, q0) = self.book.order(h).ok_or(ApplyError::UnknownOrder)?;
                    self.book.cancel(h, q0).map_err(Self::map)?;
                    self.ids.remove(id);
                }
                Ok(())
            }
            Change::Match { id, qty } => {
                let h = self.ids.get(id).ok_or(ApplyError::UnknownOrder)?;
                let (_, _, q0) = self.book.order(h).ok_or(ApplyError::UnknownOrder)?;
                if qty <= 0 {
                    return Err(ApplyError::BadSize);
                }
                let lots = self.lots_of(qty)?;
                if lots > q0 {
                    return Err(ApplyError::BadSize);
                }
                self.book.execute(h, lots).map_err(Self::map)?;
                if lots == q0 {
                    self.ids.remove(id);
                }
                Ok(())
            }
            Change::Resize { id, qty } => {
                if qty < 0 {
                    return Err(ApplyError::BadSize);
                }
                // A change for an order not on the book is a received-only order.
                if let Some(h) = self.ids.get(id) {
                    let (_, t, q0) = self.book.order(h).ok_or(ApplyError::UnknownOrder)?;
                    if qty == 0 {
                        self.book.cancel(h, q0).map_err(Self::map)?;
                        self.ids.remove(id);
                    } else {
                        let lots = self.lots_of(qty)?;
                        self.book.modify(h, t, lots).map_err(Self::map)?;
                    }
                }
                Ok(())
            }
        }
    }

    #[inline]
    fn top(&self, x: Option<(i64, u64, u32)>) -> Option<(i64, i64)> {
        x.map(|(t, q, _)| (t * self.spec.tick, q as i64 * self.spec.lot))
    }

    /// Best bid (price, total).
    pub fn best_bid(&self) -> Option<(i64, i64)> {
        self.top(self.book.best_bid())
    }
    /// Best ask (price, total).
    pub fn best_ask(&self) -> Option<(i64, i64)> {
        self.top(self.book.best_ask())
    }
    /// Levels of one side, best first. Allocates (a reporting call, not the
    /// hot path).
    pub fn levels(&self, s: Side) -> Vec<(i64, i64)> {
        let mut out: Vec<(i64, i64)> = Vec::new();
        let (tick, lot) = (self.spec.tick, self.spec.lot);
        self.book.for_each_order(qside(s), |t, _, q| {
            let px = t * tick;
            match out.last_mut() {
                Some(l) if l.0 == px => l.1 += q as i64 * lot,
                _ => out.push((px, q as i64 * lot)),
            }
        });
        out
    }
    /// Live orders by venue id: (id, side, px, qty), sorted by id.
    pub fn orders(&self) -> Vec<(u128, Side, i64, i64)> {
        let mut v: Vec<_> = self
            .ids
            .iter()
            .filter_map(|(k, h)| {
                self.book.order(h).map(|(s, t, q)| {
                    let side = if s == QSide::Buy {
                        Side::Bid
                    } else {
                        Side::Ask
                    };
                    (k, side, t * self.spec.tick, q as i64 * self.spec.lot)
                })
            })
            .collect();
        v.sort_unstable_by_key(|e| e.0);
        v
    }
    /// Number of live orders (order feeds).
    pub fn order_count(&self) -> usize {
        self.ids.len()
    }
    /// Levels equal the per-price sum of live orders (order feeds only), and
    /// every interned id maps to a live book order.
    pub fn check_invariant(&self) -> bool {
        if self.ids.len() == 0 {
            return true;
        }
        let mut b = BTreeMap::new();
        let mut a = BTreeMap::new();
        for (_, s, p, q) in self.orders() {
            *match s {
                Side::Bid => &mut b,
                Side::Ask => &mut a,
            }
            .entry(p)
            .or_insert(0) += q;
        }
        let rev = |m: BTreeMap<i64, i64>, r: bool| -> Vec<(i64, i64)> {
            let v: Vec<_> = m.into_iter().collect();
            if r { v.into_iter().rev().collect() } else { v }
        };
        self.orders().len() == self.ids.len()
            && self.book.len() == self.ids.len()
            && rev(b, true) == self.levels(Side::Bid)
            && rev(a, false) == self.levels(Side::Ask)
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
        })
        .unwrap();
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
        // Raw integer prices: one fixed-point unit per tick.
        let mut b = CryptoBook::with_spec(ProductSpec {
            tick: 1,
            ..ProductSpec::default()
        });
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

    #[test]
    fn interner_matches_a_hashmap_model() {
        use std::collections::HashMap;
        let mut t = Interner::new(64);
        let mut m: HashMap<u128, u64> = HashMap::new();
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        for _ in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // Few distinct keys so inserts, hits and deletes all collide.
            let k = ((x % 97) as u128) << 64 | (x % 89) as u128;
            match (x >> 40) % 3 {
                0 if !m.contains_key(&k) => match t.insert(k) {
                    Some(h) => {
                        assert!(h & HANDLE_BIT != 0);
                        assert!(m.values().all(|&v| v != h), "handle reused while live");
                        m.insert(k, h);
                    }
                    None => assert_eq!(m.len(), 64),
                },
                1 => {
                    t.remove(k);
                    m.remove(&k);
                }
                _ => assert_eq!(t.get(k), m.get(&k).copied()),
            }
            assert_eq!(t.len(), m.len());
        }
    }

    #[test]
    fn off_grid_and_oversize_are_errors_not_panics() {
        let mut b = CryptoBook::new(); // 0.01 tick, 1e-8 lot
        b.reset(&Snapshot {
            last: 1,
            levels: vec![(Side::Bid, 100 * SCALE, SCALE)],
            orders: vec![],
        })
        .unwrap();
        let lv = |px, qty| Change::Level {
            side: Side::Ask,
            px,
            qty,
        };
        assert_eq!(
            b.apply(&lv(100 * SCALE + 1, 1)),
            Err(ApplyError::OutOfWindow)
        );
        assert_eq!(
            b.apply(&lv(10_000 * SCALE, 1)),
            Err(ApplyError::OutOfWindow)
        );
        assert_eq!(
            b.apply(&lv(101 * SCALE, 43 * SCALE)),
            Err(ApplyError::Unrepresentable)
        );
        assert_eq!(b.apply(&lv(101 * SCALE, 42 * SCALE)), Ok(()));
        // Removing a level that is not there (even far outside) is a no-op.
        assert_eq!(b.apply(&lv(10_000 * SCALE, 0)), Ok(()));
        assert_eq!(b.best_ask(), Some((101 * SCALE, 42 * SCALE)));
        let (lo, hi) = b.window();
        assert!(lo <= 100 * SCALE && 101 * SCALE < hi);
    }
}
