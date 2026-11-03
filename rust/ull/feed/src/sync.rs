//! A/B line arbitration and snapshot synchronisation with a stale-book flag.
//!
//! # Plain idea
//!
//! Many venues publish the same stream twice (line A and line B) so that a
//! packet lost on one line can be taken from the other. The [`Arbiter`]
//! passes on each update id once, the first time either line delivers it,
//! and briefly holds updates that arrive ahead of a hole in case the other
//! line fills it. Public crypto websockets are a single line; running two
//! connections to the same stream gives the same A/B shape, and the
//! recorder and tests drive it the same way.
//!
//! The [`SyncEngine`] keeps the book honest. It starts with no book: it
//! buffers deltas, asks for a REST snapshot, drops buffered deltas the
//! snapshot already covers, checks that the first remaining delta bridges the
//! snapshot, then applies the rest in order. Any later hole in the ids marks
//! the book stale, buffers again and asks for a new snapshot. While stale,
//! quoting must stop (W26 reads [`SyncEngine::is_stale`] and pulls quotes).
//!
//! # Formal statement
//!
//! Each delta covers update ids `first..=last`. With snapshot id `L`:
//! 1. drop every delta with `last <= L`;
//! 2. the first delta applied must satisfy `first <= L + 1 <= last`;
//! 3. every later delta must have `first == previous last + 1`.
//!
//! This is the Binance procedure ("How to manage a local order book
//! correctly", Spot docs) and, with `first == last == sequence`, the Coinbase
//! `full` procedure (Coinbase Exchange docs, full channel). Invariant checked
//! in tests: whenever `is_stale()` is false, the book equals the reference
//! book at id `last_id()`.
//!
//! Arbiter: with `next` the smallest id not yet passed on, a delta with
//! `last < next` is a duplicate; one with `first <= next` is passed on and
//! unblocks any held deltas; one with `first > next` is held. If the hole is
//! still open after `window` further arrivals, the held deltas are released
//! in order and the engine sees the gap. The window is counted in messages,
//! not time, so the tests are deterministic; a production build would add a
//! time bound on the same path.
//!
//! # Worked example
//!
//! Snapshot `L = 100`. Buffered deltas `[95..=99]`, `[100..=103]`,
//! `[104..=104]`. Step 1 drops `[95..=99]`; `[100..=103]` bridges because
//! `100 <= 101 <= 103`; `[104..=104]` continues. A later `[106..=107]` skips
//! 105, so the book goes stale and a snapshot is requested.

use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};

use crate::book::{ApplyError, CryptoBook, Delta, ProductSpec, Snapshot};

/// What the caller should do after feeding the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Nothing.
    None,
    /// Fetch a REST snapshot and pass it to [`SyncEngine::on_snapshot`].
    RequestSnapshot,
}

/// Counters for tests and monitoring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncStats {
    /// Deltas applied to the book.
    pub applied: u64,
    /// Deltas dropped as already covered.
    pub dropped_old: u64,
    /// Gaps detected (each one marks the book stale).
    pub gaps: u64,
    /// Snapshots applied.
    pub snapshots: u64,
    /// Snapshots too old to bridge the buffer.
    pub snapshots_too_old: u64,
    /// Buffered deltas discarded because the buffer was full.
    pub buffer_overflow: u64,
    /// Deltas whose price fell outside the product window (each one marks
    /// the book stale; the recovery snapshot re-centres the window).
    pub out_of_window: u64,
    /// Snapshots the book could not hold (off-grid price, size too large,
    /// or more entries than slots); the engine stays stale and asks again.
    pub snapshots_rejected: u64,
    /// Snapshots ignored because they were not newer than the live book.
    pub snapshots_not_newer: u64,
}

/// Snapshot-synchronised book with a stale flag.
#[derive(Debug)]
pub struct SyncEngine {
    book: CryptoBook,
    live: bool,
    last: u64,
    requested: bool,
    buf: VecDeque<Delta>,
    cap: usize,
    /// Counters.
    pub stats: SyncStats,
}

impl SyncEngine {
    /// New engine buffering at most `cap` deltas while out of sync.
    pub fn new(cap: usize) -> Self {
        Self::with_spec(cap, ProductSpec::default())
    }

    /// New engine for a product with its own book layout.
    pub fn with_spec(cap: usize, spec: ProductSpec) -> Self {
        Self {
            book: CryptoBook::with_spec(spec),
            live: false,
            last: 0,
            requested: false,
            buf: VecDeque::with_capacity(cap),
            cap,
            stats: SyncStats::default(),
        }
    }

    /// True until a snapshot is bridged, and from any gap until the next one.
    pub fn is_stale(&self) -> bool {
        !self.live
    }
    /// Last update id reflected in the book (meaningful when not stale).
    pub fn last_id(&self) -> u64 {
        self.last
    }
    /// The book. Do not quote from it while [`Self::is_stale`] is true.
    pub fn book(&self) -> &CryptoBook {
        &self.book
    }
    /// Deltas waiting for a snapshot.
    pub fn buffered(&self) -> usize {
        self.buf.len()
    }

    fn go_stale(&mut self) -> Action {
        self.live = false;
        if self.requested {
            Action::None
        } else {
            self.requested = true;
            Action::RequestSnapshot
        }
    }

    fn push_buf(&mut self, d: Delta) {
        if self.buf.len() == self.cap {
            self.buf.pop_front();
            self.stats.buffer_overflow += 1;
        }
        self.buf.push_back(d);
    }

    fn apply(&mut self, d: &Delta) -> bool {
        for c in &d.changes {
            if let Err(e) = self.book.apply(c) {
                if e == ApplyError::OutOfWindow {
                    self.stats.out_of_window += 1;
                }
                return false;
            }
        }
        self.last = d.last;
        self.stats.applied += 1;
        true
    }

    /// Feed one delta (already arbitrated).
    pub fn on_delta(&mut self, d: Delta) -> Action {
        self.on_delta_cow(Cow::Owned(d))
    }

    /// Feed one borrowed delta. In sync and in order this does not allocate;
    /// the delta is copied only when it must be buffered (stale or a gap).
    pub fn on_delta_ref(&mut self, d: &Delta) -> Action {
        self.on_delta_cow(Cow::Borrowed(d))
    }

    fn on_delta_cow(&mut self, d: Cow<'_, Delta>) -> Action {
        if !self.live {
            self.push_buf(d.into_owned());
            return self.go_stale();
        }
        if d.last <= self.last {
            self.stats.dropped_old += 1;
            return Action::None;
        }
        if d.first == self.last + 1 && self.apply(&d) {
            return Action::None;
        }
        // Hole, overlap, or a change the book cannot apply: resync.
        self.stats.gaps += 1;
        self.push_buf(d.into_owned());
        self.go_stale()
    }

    /// Feed a snapshot (the reply to a [`Action::RequestSnapshot`]).
    pub fn on_snapshot(&mut self, s: &Snapshot) -> Action {
        self.requested = false;
        if self.live && s.last <= self.last {
            // Not newer than the book we hold (the request raced the stream,
            // or a second REST reply came back). Reloading it would rewind
            // the book and drop the levels the stream added beyond the
            // snapshot's depth. Keep our book, as cpp/ull/feed does.
            self.stats.snapshots_not_newer += 1;
            return Action::None;
        }
        if self.book.reset(s).is_err() {
            self.stats.snapshots_rejected += 1;
            self.live = false;
            return self.go_stale();
        }
        self.last = s.last;
        self.stats.snapshots += 1;
        while self.buf.front().is_some_and(|d| d.last <= s.last) {
            self.buf.pop_front();
            self.stats.dropped_old += 1;
        }
        if let Some(d) = self.buf.front()
            && d.first > s.last + 1
        {
            self.stats.snapshots_too_old += 1;
            self.live = false;
            return self.go_stale();
        }
        self.live = true;
        let mut first = true;
        while let Some(d) = self.buf.pop_front() {
            let ok = (if first {
                d.first <= self.last + 1
            } else {
                d.first == self.last + 1
            }) && self.apply(&d);
            first = false;
            if !ok {
                self.stats.gaps += 1;
                self.buf.push_front(d);
                return self.go_stale();
            }
        }
        Action::None
    }
}

/// Counters for the arbiter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArbStats {
    /// Deltas passed on, per line (A = 0, B = 1).
    pub won: [u64; 2],
    /// Duplicates discarded.
    pub duplicates: u64,
    /// Holes given up on (released to the engine as a gap).
    pub gaps_released: u64,
}

/// First-arrival A/B arbitration over update ids.
#[derive(Debug)]
pub struct Arbiter {
    next: Option<u64>,
    held: BTreeMap<u64, Delta>,
    window: usize,
    waited: usize,
    /// Counters.
    pub stats: ArbStats,
}

impl Arbiter {
    /// Hold out-of-order deltas for at most `window` further arrivals.
    pub fn new(window: usize) -> Self {
        Self {
            next: None,
            held: BTreeMap::new(),
            window,
            waited: 0,
            stats: ArbStats::default(),
        }
    }

    fn emit(&mut self, d: Delta, out: &mut Vec<Delta>) {
        self.next = Some(d.last + 1);
        out.push(d);
    }

    fn drain(&mut self, out: &mut Vec<Delta>) {
        while let Some(e) = self.held.first_entry() {
            let next = self.next.unwrap_or(0);
            if *e.key() > next {
                break;
            }
            let d = e.remove();
            if d.last < next {
                self.stats.duplicates += 1;
            } else {
                self.emit(d, out);
            }
        }
        if self.held.is_empty() {
            self.waited = 0;
        }
    }

    /// Fast path for a delta the caller still owns: when nothing is held and
    /// the delta is either next in order or a duplicate, update the state
    /// exactly as [`Self::on`] would and return `Some(true)` (pass it on) or
    /// `Some(false)` (duplicate, drop it). Otherwise return `None` and change
    /// nothing; the caller then hands an owned copy to [`Self::on`].
    #[inline]
    pub fn try_pass(&mut self, line: usize, first: u64, last: u64) -> Option<bool> {
        if !self.held.is_empty() {
            return None;
        }
        match self.next {
            Some(next) if last < next => {
                self.stats.duplicates += 1;
                Some(false)
            }
            Some(next) if first > next => None,
            _ => {
                self.stats.won[line & 1] += 1;
                self.next = Some(last + 1);
                Some(true)
            }
        }
    }

    /// Offer a delta from `line` (0 = A, 1 = B); in-order output goes to `out`.
    pub fn on(&mut self, line: usize, d: Delta, out: &mut Vec<Delta>) {
        let Some(next) = self.next else {
            self.stats.won[line & 1] += 1;
            self.emit(d, out);
            return;
        };
        if d.last < next || self.held.contains_key(&d.first) {
            self.stats.duplicates += 1;
        } else if d.first <= next {
            self.stats.won[line & 1] += 1;
            self.emit(d, out);
            self.drain(out);
        } else {
            self.stats.won[line & 1] += 1;
            self.held.insert(d.first, d);
        }
        if !self.held.is_empty() {
            self.waited += 1;
            if self.waited > self.window {
                // Give up on the hole: release the earliest held delta; the
                // engine sees the gap, marks the book stale and resyncs.
                self.stats.gaps_released += 1;
                if let Some((_, d)) = self.held.pop_first() {
                    self.emit(d, out);
                }
                self.drain(out);
                self.waited = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::{Change, ProductSpec, Side};

    // Unit tests use raw integer prices: one fixed-point unit per tick.
    const UNIT: ProductSpec = ProductSpec {
        tick: 1,
        lot: 1,
        n_levels: 1 << 12,
        max_orders: 1 << 10,
    };

    fn d(first: u64, last: u64) -> Delta {
        Delta {
            first,
            last,
            changes: vec![Change::Level {
                side: Side::Bid,
                px: last as i64,
                qty: 1,
            }],
        }
    }

    #[test]
    fn worked_example() {
        let mut e = SyncEngine::with_spec(16, UNIT);
        assert_eq!(e.on_delta(d(95, 99)), Action::RequestSnapshot);
        assert_eq!(e.on_delta(d(100, 103)), Action::None);
        e.on_delta(d(104, 104));
        assert!(e.is_stale());
        assert_eq!(
            e.on_snapshot(&Snapshot {
                last: 100,
                ..Default::default()
            }),
            Action::None
        );
        assert!(!e.is_stale());
        assert_eq!(e.last_id(), 104);
        assert_eq!(e.on_delta(d(106, 107)), Action::RequestSnapshot);
        assert!(e.is_stale());
        assert_eq!(e.stats.gaps, 1);
    }

    #[test]
    fn snapshot_too_old_asks_again() {
        let mut e = SyncEngine::with_spec(16, UNIT);
        e.on_delta(d(50, 51));
        assert_eq!(
            e.on_snapshot(&Snapshot {
                last: 40,
                ..Default::default()
            }),
            Action::RequestSnapshot
        );
        assert!(e.is_stale());
        assert_eq!(
            e.on_snapshot(&Snapshot {
                last: 50,
                ..Default::default()
            }),
            Action::None
        );
        assert_eq!(e.last_id(), 51);
    }

    #[test]
    fn arbiter_fills_hole_from_other_line() {
        let mut a = Arbiter::new(4);
        let mut out = vec![];
        a.on(0, d(1, 1), &mut out);
        a.on(0, d(3, 3), &mut out); // A lost 2
        a.on(1, d(1, 1), &mut out); // dup
        a.on(1, d(2, 2), &mut out); // B fills it
        a.on(1, d(3, 3), &mut out); // dup
        let ids: Vec<u64> = out.iter().map(|x| x.first).collect();
        assert_eq!(ids, [1, 2, 3]);
        assert_eq!(a.stats.duplicates, 2);
    }

    #[test]
    fn arbiter_releases_after_window() {
        let mut a = Arbiter::new(2);
        let mut out = vec![];
        a.on(0, d(1, 1), &mut out);
        for i in 3..=5 {
            a.on(0, d(i, i), &mut out);
        }
        let ids: Vec<u64> = out.iter().map(|x| x.first).collect();
        assert_eq!(ids, [1, 3, 4, 5]);
        assert_eq!(a.stats.gaps_released, 1);
    }

    #[test]
    fn snapshot_not_newer_than_live_book_is_ignored() {
        let lv = |px: i64, qty: i64| Change::Level {
            side: Side::Bid,
            px,
            qty,
        };
        let mut e = SyncEngine::with_spec(16, UNIT);
        e.on_delta(Delta {
            first: 11,
            last: 11,
            changes: vec![lv(5, 1)],
        });
        let snap = |last: u64| Snapshot {
            last,
            levels: vec![(Side::Bid, 5, 1)],
            orders: vec![],
        };
        assert_eq!(e.on_snapshot(&snap(10)), Action::None);
        assert!(!e.is_stale());
        // A level the snapshot (limited depth) does not carry.
        e.on_delta(Delta {
            first: 12,
            last: 12,
            changes: vec![lv(3, 7)],
        });
        let before = e.book().levels(Side::Bid);
        assert_eq!(e.on_snapshot(&snap(12)), Action::None); // same id: not newer
        assert_eq!(e.on_snapshot(&snap(11)), Action::None); // older
        assert_eq!(e.book().levels(Side::Bid), before);
        assert_eq!(e.stats.snapshots_not_newer, 2);
        assert_eq!(e.last_id(), 12);
        assert!(!e.is_stale());
    }
}
