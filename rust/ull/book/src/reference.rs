//! Naive reference book: the test oracle for `Book`.
//!
//! Written for obviousness, not speed: a `BTreeMap` per side from price to a
//! `VecDeque` of order ids in time priority, plus a `HashMap` from id to
//! `(side, price, qty)`. It applies the same priority rule as `Book`.
//!
//! The plan names the W25 reference book (another lane) as the oracle. Until
//! it merges this module stands in; at integration the W25 book should be run
//! on the same generated days and compared with both (handback item).

use crate::BookError;
use qp_ull_qpbf::{Msg, Side};
use std::collections::{BTreeMap, HashMap, VecDeque};

/// Reference book.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RefBook {
    bids: BTreeMap<i64, VecDeque<u64>>,
    asks: BTreeMap<i64, VecDeque<u64>>,
    orders: HashMap<u64, (Side, i64, u32)>,
}

impl RefBook {
    fn side(&mut self, s: Side) -> &mut BTreeMap<i64, VecDeque<u64>> {
        match s {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        }
    }

    fn take_out(&mut self, id: u64) {
        let (s, p, _) = self.orders[&id];
        let lv = self.side(s);
        let q = lv.get_mut(&p).expect("level");
        q.retain(|&x| x != id);
        if q.is_empty() {
            lv.remove(&p);
        }
    }

    /// Applies one message with the same semantics and errors as `Book`
    /// (no window, so no `OutOfRange`/`Full`).
    pub fn apply(&mut self, m: &Msg) -> Result<(), BookError> {
        match *m {
            Msg::Add {
                order_id,
                side,
                price,
                qty,
                ..
            } => {
                if qty == 0 {
                    return Err(BookError::BadQty);
                }
                if self.orders.contains_key(&order_id) {
                    return Err(BookError::DuplicateOrder(order_id));
                }
                self.orders.insert(order_id, (side, price, qty));
                self.side(side)
                    .entry(price)
                    .or_default()
                    .push_back(order_id);
            }
            Msg::Modify {
                order_id,
                price,
                qty,
                ..
            } => {
                let (s, p, q) = *self
                    .orders
                    .get(&order_id)
                    .ok_or(BookError::UnknownOrder(order_id))?;
                if qty == 0 {
                    return Err(BookError::BadQty);
                }
                if !(price == p && qty <= q) {
                    self.take_out(order_id);
                    self.side(s).entry(price).or_default().push_back(order_id);
                }
                self.orders.insert(order_id, (s, price, qty));
            }
            Msg::Cancel { order_id, qty, .. } | Msg::Execute { order_id, qty, .. } => {
                let (_, _, q) = *self
                    .orders
                    .get(&order_id)
                    .ok_or(BookError::UnknownOrder(order_id))?;
                if qty == 0 || qty > q {
                    return Err(BookError::BadQty);
                }
                if qty == q {
                    self.take_out(order_id);
                    self.orders.remove(&order_id);
                } else {
                    self.orders.get_mut(&order_id).expect("live").2 -= qty;
                }
            }
            Msg::Snapshot {
                order_id,
                side,
                price,
                qty,
                ..
            } => {
                if order_id != 0 {
                    return self.apply(&Msg::Add {
                        instrument: 0,
                        ts_ns: 0,
                        order_id,
                        side,
                        price,
                        qty,
                    });
                }
            }
            Msg::Trade { .. } | Msg::Heartbeat { .. } => {}
        }
        Ok(())
    }

    /// Empties the book (receiver starts applying a snapshot).
    pub fn clear(&mut self) {
        *self = RefBook::default();
    }

    /// Every order on `side` in priority order as `(price, id, qty)`.
    pub fn orders(&self, side: Side) -> Vec<(i64, u64, u32)> {
        let one = |(p, q): (&i64, &VecDeque<u64>)| {
            q.iter()
                .map(|id| (*p, *id, self.orders[id].2))
                .collect::<Vec<_>>()
        };
        match side {
            Side::Buy => self.bids.iter().rev().flat_map(one).collect(),
            Side::Sell => self.asks.iter().flat_map(one).collect(),
        }
    }

    /// Best `(price, qty, count)` on a side.
    pub fn best(&self, side: Side) -> Option<(i64, u64, u32)> {
        let lv = match side {
            Side::Buy => self.bids.iter().next_back(),
            Side::Sell => self.asks.iter().next(),
        }?;
        let qty = lv.1.iter().map(|id| self.orders[id].2 as u64).sum();
        Some((*lv.0, qty, lv.1.len() as u32))
    }
}
