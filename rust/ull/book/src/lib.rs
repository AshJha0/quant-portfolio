//! L3 (order-by-order) book on flat price-indexed arrays.
//!
//! Plain idea: a price is an integer tick, so the book can be an array with
//! one slot per tick inside a fixed window. Each slot holds the level's total
//! quantity and a first-in-first-out list of the orders resting there. Orders
//! live in a pre-allocated pool and are found by id through a pre-allocated
//! open-addressing table. After `Book::new` nothing allocates: add, modify,
//! cancel and execute are array and pointer updates.
//!
//! Formal statement: for a window `[base, base + n_levels)`, level `p - base`
//! of side `s` holds exactly the live orders of side `s` at price `p`, linked
//! in priority order, and `qty` equals the sum of their remaining quantities.
//! `best_bid` is the highest non-empty bid index and `best_ask` the lowest
//! non-empty ask index. Priority rule (SPEC CHOICE, shared with the
//! generator): a Modify keeps the order's place only when the price is
//! unchanged and the quantity does not increase; otherwise the order moves to
//! the back of its (new) level. An Execute that leaves zero removes the order.
//!
//! Worked example: add bid 1 (10 @ 100), add bid 2 (5 @ 100), add ask 3
//! (7 @ 102). Best bid is 100 with 15 in 2 orders; best ask is 102. Execute 4
//! on order 1: level 100 has 11, order 1 still first. Modify order 1 to 20 @
//! 100: it moves behind order 2. Cancel 2 and 1: best bid becomes empty. The
//! unit test `worked_example` walks exactly this sequence.
//!
//! The flat-array layout follows the common practice described for
//! price-level books in, for example, Drepper (2007) on keeping hot data
//! contiguous; the window is fixed at construction and a price outside it is
//! an error, not a resize (a resize would allocate on the hot path).

#![warn(missing_docs)]

pub mod reference;

use qp_ull_qpbf::{Msg, Side};

const NIL: u32 = u32::MAX;

/// Book rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookError {
    /// Order id not live.
    UnknownOrder(u64),
    /// Add of an id that is already live.
    DuplicateOrder(u64),
    /// Price outside the window.
    OutOfRange(i64),
    /// Order pool exhausted.
    Full,
    /// Zero quantity, or an execute larger than the remaining quantity.
    BadQty,
}

#[derive(Clone, Copy, Debug)]
struct Level {
    qty: u64,
    count: u32,
    head: u32,
    tail: u32,
}

const EMPTY_LEVEL: Level = Level {
    qty: 0,
    count: 0,
    head: NIL,
    tail: NIL,
};

#[derive(Clone, Copy, Debug)]
struct Order {
    id: u64,
    price_ix: u32,
    qty: u32,
    side: Side,
    prev: u32,
    next: u32,
}

/// Fixed-capacity id -> slot map; linear probing with backward-shift delete
/// (Knuth, TAOCP vol. 3, 6.4, Algorithm R), so there are no tombstones.
#[derive(Debug)]
struct IdMap {
    keys: Box<[u64]>, // 0 = empty; order id 0 is reserved (SPEC CHOICE)
    vals: Box<[u32]>,
    mask: usize,
}

impl IdMap {
    fn new(cap: usize) -> Self {
        let n = (cap * 2).max(2).next_power_of_two();
        IdMap {
            keys: vec![0; n].into_boxed_slice(),
            vals: vec![0; n].into_boxed_slice(),
            mask: n - 1,
        }
    }
    #[inline]
    fn home(&self, k: u64) -> usize {
        (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & self.mask
    }
    #[inline]
    fn get(&self, k: u64) -> Option<u32> {
        let mut i = self.home(k);
        loop {
            let kk = self.keys[i];
            if kk == k {
                return Some(self.vals[i]);
            }
            if kk == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline]
    fn insert(&mut self, k: u64, v: u32) {
        let mut i = self.home(k);
        while self.keys[i] != 0 {
            i = (i + 1) & self.mask;
        }
        self.keys[i] = k;
        self.vals[i] = v;
    }
    #[inline]
    fn remove(&mut self, k: u64) {
        let mut i = self.home(k);
        while self.keys[i] != k {
            if self.keys[i] == 0 {
                return;
            }
            i = (i + 1) & self.mask;
        }
        // Backward-shift: pull later entries of the probe run into the hole.
        let mut hole = i;
        let mut j = i;
        loop {
            j = (j + 1) & self.mask;
            let kj = self.keys[j];
            if kj == 0 {
                break;
            }
            let h = self.home(kj);
            // Move kj into the hole unless its home lies cyclically in (hole, j].
            let in_range = if hole <= j {
                hole < h && h <= j
            } else {
                hole < h || h <= j
            };
            if !in_range {
                self.keys[hole] = kj;
                self.vals[hole] = self.vals[j];
                hole = j;
            }
        }
        self.keys[hole] = 0;
    }
    fn clear(&mut self) {
        self.keys.fill(0);
    }
}

/// Allocation-free L3 book for one instrument.
#[derive(Debug)]
pub struct Book {
    base: i64,
    bids: Box<[Level]>,
    asks: Box<[Level]>,
    orders: Box<[Order]>,
    free: u32,
    live: u32,
    ids: IdMap,
    best_bid: u32, // NIL when empty
    best_ask: u32,
}

impl Book {
    /// A book for prices `[base, base + n_levels)` holding up to
    /// `max_orders` live orders. All memory is allocated here.
    pub fn new(base: i64, n_levels: u32, max_orders: u32) -> Self {
        assert!(n_levels > 0 && n_levels < NIL && max_orders > 0 && max_orders < NIL);
        let orders: Box<[Order]> = (0..max_orders)
            .map(|i| Order {
                id: 0,
                price_ix: 0,
                qty: 0,
                side: Side::Buy,
                prev: NIL,
                next: if i + 1 < max_orders { i + 1 } else { NIL },
            })
            .collect();
        Book {
            base,
            bids: vec![EMPTY_LEVEL; n_levels as usize].into_boxed_slice(),
            asks: vec![EMPTY_LEVEL; n_levels as usize].into_boxed_slice(),
            orders,
            free: 0,
            live: 0,
            ids: IdMap::new(max_orders as usize),
            best_bid: NIL,
            best_ask: NIL,
        }
    }

    /// Live order count.
    pub fn len(&self) -> usize {
        self.live as usize
    }

    /// True when no order is live.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    #[inline]
    fn ix(&self, price: i64) -> Result<u32, BookError> {
        let d = price.wrapping_sub(self.base);
        if d < 0 || d >= self.bids.len() as i64 {
            return Err(BookError::OutOfRange(price));
        }
        Ok(d as u32)
    }

    #[inline]
    fn levels(&mut self, side: Side) -> &mut [Level] {
        match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        }
    }

    /// Links slot `s` at the back of its level and updates the best index.
    #[inline]
    fn link_back(&mut self, s: u32) {
        let Order {
            side,
            price_ix,
            qty,
            ..
        } = self.orders[s as usize];
        let lv = &mut self.levels(side)[price_ix as usize];
        let old_tail = lv.tail;
        lv.qty += qty as u64;
        lv.count += 1;
        lv.tail = s;
        if old_tail == NIL {
            lv.head = s;
        }
        if old_tail != NIL {
            self.orders[old_tail as usize].next = s;
        }
        let o = &mut self.orders[s as usize];
        o.prev = old_tail;
        o.next = NIL;
        match side {
            Side::Buy if self.best_bid == NIL || price_ix > self.best_bid => {
                self.best_bid = price_ix
            }
            Side::Sell if self.best_ask == NIL || price_ix < self.best_ask => {
                self.best_ask = price_ix
            }
            _ => {}
        }
    }

    /// Unlinks slot `s` from its level and repairs the best index.
    #[inline]
    fn unlink(&mut self, s: u32) {
        let Order {
            side,
            price_ix,
            qty,
            prev,
            next,
            ..
        } = self.orders[s as usize];
        if prev != NIL {
            self.orders[prev as usize].next = next;
        }
        if next != NIL {
            self.orders[next as usize].prev = prev;
        }
        let lv = &mut self.levels(side)[price_ix as usize];
        if prev == NIL {
            lv.head = next;
        }
        if next == NIL {
            lv.tail = prev;
        }
        lv.qty -= qty as u64;
        lv.count -= 1;
        if lv.count == 0 {
            self.repair_best(side, price_ix);
        }
    }

    /// Level `ix` on `side` just emptied; if it was best, scan outward to the
    /// next non-empty level (bounded by the window).
    fn repair_best(&mut self, side: Side, ix: u32) {
        match side {
            Side::Buy if self.best_bid == ix => {
                self.best_bid = (0..ix)
                    .rev()
                    .find(|&i| self.bids[i as usize].count > 0)
                    .unwrap_or(NIL);
            }
            Side::Sell if self.best_ask == ix => {
                self.best_ask = (ix + 1..self.asks.len() as u32)
                    .find(|&i| self.asks[i as usize].count > 0)
                    .unwrap_or(NIL);
            }
            _ => {}
        }
    }

    fn release(&mut self, s: u32) {
        let id = self.orders[s as usize].id;
        self.ids.remove(id);
        self.orders[s as usize].next = self.free;
        self.free = s;
        self.live -= 1;
    }

    /// Adds a resting order.
    pub fn add(&mut self, id: u64, side: Side, price: i64, qty: u32) -> Result<(), BookError> {
        if qty == 0 || id == 0 {
            return Err(if qty == 0 {
                BookError::BadQty
            } else {
                BookError::UnknownOrder(0)
            });
        }
        let ix = self.ix(price)?;
        if self.ids.get(id).is_some() {
            return Err(BookError::DuplicateOrder(id));
        }
        let s = self.free;
        if s == NIL {
            return Err(BookError::Full);
        }
        self.free = self.orders[s as usize].next;
        self.orders[s as usize] = Order {
            id,
            price_ix: ix,
            qty,
            side,
            prev: NIL,
            next: NIL,
        };
        self.ids.insert(id, s);
        self.live += 1;
        self.link_back(s);
        Ok(())
    }

    /// Replaces price and quantity; see the module priority rule.
    pub fn modify(&mut self, id: u64, price: i64, qty: u32) -> Result<(), BookError> {
        if qty == 0 {
            return Err(BookError::BadQty);
        }
        let s = self.ids.get(id).ok_or(BookError::UnknownOrder(id))?;
        let ix = self.ix(price)?;
        let o = self.orders[s as usize];
        if ix == o.price_ix && qty <= o.qty {
            let side = o.side;
            self.levels(side)[ix as usize].qty -= (o.qty - qty) as u64;
            self.orders[s as usize].qty = qty;
        } else {
            self.unlink(s);
            let o = &mut self.orders[s as usize];
            o.price_ix = ix;
            o.qty = qty;
            self.link_back(s);
        }
        Ok(())
    }

    /// Owner removes `qty` from an order (spec section 5 rule 3); the order
    /// is removed when nothing remains. Same arithmetic as `execute`.
    pub fn cancel(&mut self, id: u64, qty: u32) -> Result<(), BookError> {
        self.reduce(id, qty)
    }

    /// Fills `qty` of a resting order at its price.
    pub fn execute(&mut self, id: u64, qty: u32) -> Result<(), BookError> {
        self.reduce(id, qty)
    }

    #[inline]
    fn reduce(&mut self, id: u64, qty: u32) -> Result<(), BookError> {
        let s = self.ids.get(id).ok_or(BookError::UnknownOrder(id))?;
        let o = self.orders[s as usize];
        if qty == 0 || qty > o.qty {
            return Err(BookError::BadQty);
        }
        if qty == o.qty {
            self.unlink(s);
            self.release(s);
        } else {
            self.orders[s as usize].qty -= qty;
            let side = o.side;
            self.levels(side)[o.price_ix as usize].qty -= qty as u64;
        }
        Ok(())
    }

    /// Removes every order. O(window + id table); used by snapshots only.
    pub fn clear(&mut self) {
        self.bids.fill(EMPTY_LEVEL);
        self.asks.fill(EMPTY_LEVEL);
        let n = self.orders.len() as u32;
        for (i, o) in self.orders.iter_mut().enumerate() {
            o.next = if (i as u32) + 1 < n {
                i as u32 + 1
            } else {
                NIL
            };
        }
        self.free = 0;
        self.live = 0;
        self.ids.clear();
        self.best_bid = NIL;
        self.best_ask = NIL;
    }

    /// Applies one QPBF message. Trade and Heartbeat do not change the book.
    /// A Snapshot message adds its order (the end marker, order id 0, adds
    /// nothing). Clearing the book when a snapshot starts is the receiver's
    /// job (call `clear`), because the wire has no first-message flag.
    #[inline]
    pub fn apply(&mut self, m: &Msg) -> Result<(), BookError> {
        match *m {
            Msg::Add {
                order_id,
                side,
                price,
                qty,
                ..
            } => self.add(order_id, side, price, qty),
            Msg::Modify {
                order_id,
                price,
                qty,
                ..
            } => self.modify(order_id, price, qty),
            Msg::Cancel { order_id, qty, .. } => self.cancel(order_id, qty),
            Msg::Execute { order_id, qty, .. } => self.execute(order_id, qty),
            Msg::Snapshot {
                order_id,
                side,
                price,
                qty,
                ..
            } => {
                if order_id == 0 {
                    Ok(())
                } else {
                    self.add(order_id, side, price, qty)
                }
            }
            Msg::Trade { .. } | Msg::Heartbeat { .. } => Ok(()),
        }
    }

    fn price(&self, ix: u32) -> i64 {
        self.base + ix as i64
    }

    /// Best bid `(price, total qty, order count)`.
    pub fn best_bid(&self) -> Option<(i64, u64, u32)> {
        (self.best_bid != NIL).then(|| {
            let l = self.bids[self.best_bid as usize];
            (self.price(self.best_bid), l.qty, l.count)
        })
    }

    /// Best ask `(price, total qty, order count)`.
    pub fn best_ask(&self) -> Option<(i64, u64, u32)> {
        (self.best_ask != NIL).then(|| {
            let l = self.asks[self.best_ask as usize];
            (self.price(self.best_ask), l.qty, l.count)
        })
    }

    /// `(qty, count)` at a price; `(0, 0)` outside the window.
    pub fn level(&self, side: Side, price: i64) -> (u64, u32) {
        match self.ix(price) {
            Ok(i) => {
                let l = match side {
                    Side::Buy => self.bids[i as usize],
                    Side::Sell => self.asks[i as usize],
                };
                (l.qty, l.count)
            }
            Err(_) => (0, 0),
        }
    }

    /// `(side, price, remaining qty)` of a live order.
    pub fn order(&self, id: u64) -> Option<(Side, i64, u32)> {
        let s = self.ids.get(id)?;
        let o = self.orders[s as usize];
        Some((o.side, self.price(o.price_ix), o.qty))
    }

    /// Visits every live order on `side` in priority order (best price
    /// first, then time) as `(price, id, qty)`. Does not allocate.
    pub fn for_each_order(&self, side: Side, mut f: impl FnMut(i64, u64, u32)) {
        let (lv, start) = match side {
            Side::Buy => (&self.bids, self.best_bid),
            Side::Sell => (&self.asks, self.best_ask),
        };
        if start == NIL {
            return;
        }
        let mut visit = |i: u32| {
            let mut s = lv[i as usize].head;
            while s != NIL {
                let o = self.orders[s as usize];
                f(self.price(i), o.id, o.qty);
                s = o.next;
            }
        };
        match side {
            Side::Buy => (0..=start).rev().for_each(&mut visit),
            Side::Sell => (start..lv.len() as u32).for_each(&mut visit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bids(b: &Book) -> Vec<(i64, u64, u32)> {
        let mut v = Vec::new();
        b.for_each_order(Side::Buy, |p, id, q| v.push((p, id, q)));
        v
    }

    #[test]
    fn worked_example() {
        let mut b = Book::new(0, 1000, 16);
        b.add(1, Side::Buy, 100, 10).unwrap();
        b.add(2, Side::Buy, 100, 5).unwrap();
        b.add(3, Side::Sell, 102, 7).unwrap();
        assert_eq!(b.best_bid(), Some((100, 15, 2)));
        assert_eq!(b.best_ask(), Some((102, 7, 1)));
        b.execute(1, 4).unwrap();
        assert_eq!(b.best_bid(), Some((100, 11, 2)));
        assert_eq!(bids(&b), vec![(100, 1, 6), (100, 2, 5)]);
        b.modify(1, 100, 20).unwrap();
        assert_eq!(bids(&b), vec![(100, 2, 5), (100, 1, 20)]);
        b.modify(1, 100, 3).unwrap(); // decrease keeps place
        assert_eq!(bids(&b), vec![(100, 2, 5), (100, 1, 3)]);
        b.cancel(2, 5).unwrap();
        b.cancel(1, 3).unwrap();
        assert_eq!(b.best_bid(), None);
        assert_eq!(b.best_ask(), Some((102, 7, 1)));
        assert!(b.order(1).is_none());
    }

    /// `docs/QPBF.md` section 5 worked example.
    #[test]
    fn spec_section_5_worked_example() {
        let mut b = Book::new(0, 1000, 16);
        b.add(1, Side::Buy, 103, 10).unwrap();
        b.add(2, Side::Buy, 103, 5).unwrap();
        b.add(3, Side::Sell, 105, 7).unwrap();
        assert_eq!(b.best_bid(), Some((103, 15, 2)));
        assert_eq!(b.best_ask(), Some((105, 7, 1)));
        b.modify(1, 103, 4).unwrap();
        assert_eq!(bids(&b), vec![(103, 1, 4), (103, 2, 5)]);
        b.modify(1, 103, 6).unwrap();
        assert_eq!(bids(&b), vec![(103, 2, 5), (103, 1, 6)]);
        b.execute(2, 5).unwrap();
        assert_eq!(bids(&b), vec![(103, 1, 6)]);
    }

    #[test]
    fn best_repairs_across_gaps() {
        let mut b = Book::new(0, 100, 8);
        b.add(1, Side::Buy, 10, 1).unwrap();
        b.add(2, Side::Buy, 50, 1).unwrap();
        b.add(3, Side::Sell, 60, 1).unwrap();
        b.add(4, Side::Sell, 90, 1).unwrap();
        b.cancel(2, 1).unwrap();
        b.execute(3, 1).unwrap();
        assert_eq!(b.best_bid().unwrap().0, 10);
        assert_eq!(b.best_ask().unwrap().0, 90);
    }

    #[test]
    fn rejects() {
        let mut b = Book::new(100, 10, 1);
        assert_eq!(b.add(1, Side::Buy, 99, 1), Err(BookError::OutOfRange(99)));
        assert_eq!(b.add(1, Side::Buy, 110, 1), Err(BookError::OutOfRange(110)));
        assert_eq!(b.add(1, Side::Buy, 100, 0), Err(BookError::BadQty));
        b.add(1, Side::Buy, 100, 5).unwrap();
        assert_eq!(
            b.add(1, Side::Buy, 101, 1),
            Err(BookError::DuplicateOrder(1))
        );
        assert_eq!(b.add(2, Side::Buy, 101, 1), Err(BookError::Full));
        assert_eq!(b.execute(1, 6), Err(BookError::BadQty));
        assert_eq!(b.cancel(9, 1), Err(BookError::UnknownOrder(9)));
        assert_eq!(b.modify(1, 200, 1), Err(BookError::OutOfRange(200)));
        assert_eq!(b.order(1), Some((Side::Buy, 100, 5)));
    }

    #[test]
    fn idmap_backward_shift_keeps_every_key_findable() {
        let mut m = IdMap::new(64);
        let mut live = std::collections::HashMap::new();
        let mut r = qp_ull_qpbf::generator::Rng::new(5);
        for step in 0..200_000u32 {
            let k = 1 + r.below(100);
            if live.len() < 60 && !live.contains_key(&k) && r.below(2) == 0 {
                m.insert(k, step);
                live.insert(k, step);
            } else if live.remove(&k).is_some() {
                m.remove(k);
            }
            if step % 97 == 0 {
                for k in 1..=100 {
                    assert_eq!(m.get(k), live.get(&k).copied());
                }
            }
        }
    }
}
