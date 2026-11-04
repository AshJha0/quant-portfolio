//! W30-rs part 2: crypto feed handler.
//!
//! Pipeline for one product: raw bytes from line A or B (optionally written
//! by the [`recorder`]) -> venue parser ([`venue`]) -> [`sync::Arbiter`] ->
//! [`sync::SyncEngine`] -> [`book::CryptoBook`] with a stale flag.
//! [`Handler`] wires those together; [`generator`] produces seeded
//! Binance-shaped days with a reference book for the tests and the bench.
//!
//! Hot path: the book is the part-1 flat-array book behind a per-product
//! [`book::ProductSpec`] (tick window, lot size, UUID interning), and Binance
//! diff-depth events are read by the non-allocating decoder in [`fast`]. In
//! sync and in order, a Binance message is decoded, arbitrated and applied
//! with zero heap allocations (tests/no_alloc.rs counts them). Coinbase
//! `level2` updates have the same kind of decoder,
//! [`fast::coinbase_level2_into`], also counted at zero. Coinbase `full`
//! still goes through the generic JSON reader, which allocates; its book
//! operations do not.
//!
//! No network code lives here and no default test opens a socket. Recording a
//! live day needs a websocket client with TLS, which this std-only crate does
//! not carry; the recorder format is fixed so a separate tool can write it.

#![warn(missing_docs)]

pub mod book;
pub mod fast;
pub mod generator;
pub mod json;
pub mod recorder;
pub mod sync;
pub mod venue;

use book::{Delta, ProductSpec};
use fast::Fast;
use sync::{Action, Arbiter, SyncEngine};
use venue::{Msg, ParseError};

/// Which venue parser to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Venue {
    /// Binance spot diff-depth (`<symbol>@depth` / `@depth@100ms`).
    BinanceDiffDepth,
    /// Coinbase Exchange `full` channel (sequenced, order by order).
    CoinbaseFull,
}

/// One product's feed handler: arbitration, sync and book.
#[derive(Debug)]
pub struct Handler {
    venue: Venue,
    arb: Arbiter,
    /// The synchronised book.
    pub engine: SyncEngine,
    out: Vec<Delta>,
    scratch: Delta,
    /// Messages that failed to parse (counted, then skipped).
    pub parse_errors: u64,
    /// Stream messages decoded by the non-allocating fast path.
    pub fast_decoded: u64,
}

/// Room for this many changes per message on the fast path; a larger
/// message falls back to the generic parser (it still applies correctly).
pub const FAST_CHANGES: usize = 4096;

impl Handler {
    /// `window`: arbitration hold, in messages; `cap`: resync buffer size.
    pub fn new(venue: Venue, window: usize, cap: usize) -> Self {
        Self::with_spec(venue, window, cap, ProductSpec::default())
    }

    /// As [`Self::new`], with the product's book layout.
    pub fn with_spec(venue: Venue, window: usize, cap: usize, spec: ProductSpec) -> Self {
        Self {
            venue,
            arb: Arbiter::new(window),
            engine: SyncEngine::with_spec(cap, spec),
            out: Vec::with_capacity(64),
            scratch: Delta {
                first: 0,
                last: 0,
                changes: Vec::with_capacity(FAST_CHANGES),
            },
            parse_errors: 0,
            fast_decoded: 0,
        }
    }

    fn parse(&self, raw: &[u8]) -> Result<Msg, ParseError> {
        match self.venue {
            Venue::BinanceDiffDepth => venue::binance(raw),
            Venue::CoinbaseFull => venue::coinbase_full(raw),
        }
    }

    /// Feed one stream message from `line`. Returns `RequestSnapshot` when
    /// the caller should fetch the REST book.
    pub fn on_stream(&mut self, line: usize, raw: &[u8]) -> Action {
        if self.venue == Venue::BinanceDiffDepth
            && fast::binance_into(raw, &mut self.scratch) == Fast::Delta
        {
            self.fast_decoded += 1;
            let (first, last) = (self.scratch.first, self.scratch.last);
            return match self.arb.try_pass(line, first, last) {
                Some(true) => self.engine.on_delta_ref(&self.scratch),
                Some(false) => Action::None,
                None => {
                    let d = self.scratch.clone();
                    self.arbitrate(line, d)
                }
            };
        }
        let d = match self.parse(raw) {
            Ok(Msg::Delta(d)) => d,
            Ok(_) => return Action::None,
            Err(_) => {
                self.parse_errors += 1;
                return Action::None;
            }
        };
        self.arbitrate(line, d)
    }

    fn arbitrate(&mut self, line: usize, d: Delta) -> Action {
        self.arb.on(line, d, &mut self.out);
        let mut act = Action::None;
        let mut out = std::mem::take(&mut self.out);
        for d in out.drain(..) {
            if self.engine.on_delta(d) == Action::RequestSnapshot {
                act = Action::RequestSnapshot;
            }
        }
        self.out = out;
        act
    }

    /// Feed a REST snapshot reply.
    pub fn on_snapshot(&mut self, raw: &[u8]) -> Action {
        match self.parse(raw) {
            Ok(Msg::Snapshot(s)) => self.engine.on_snapshot(&s),
            _ => {
                self.parse_errors += 1;
                Action::RequestSnapshot
            }
        }
    }

    /// Arbitration counters.
    pub fn arb_stats(&self) -> sync::ArbStats {
        self.arb.stats
    }

    /// True while quotes must be pulled.
    pub fn is_stale(&self) -> bool {
        self.engine.is_stale()
    }
}
