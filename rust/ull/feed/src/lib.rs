//! W30-rs part 2: crypto feed handler.
//!
//! Pipeline for one product: raw bytes from line A or B (optionally written
//! by the [`recorder`]) -> venue parser ([`venue`]) -> [`sync::Arbiter`] ->
//! [`sync::SyncEngine`] -> [`book::CryptoBook`] with a stale flag.
//! [`Handler`] wires those together; [`generator`] produces seeded
//! Binance-shaped days with a reference book for the tests and the bench.
//!
//! No network code lives here and no default test opens a socket. Recording a
//! live day needs a websocket client with TLS, which this std-only crate does
//! not carry; the recorder format is fixed so a separate tool can write it.

#![warn(missing_docs)]

pub mod book;
pub mod generator;
pub mod json;
pub mod recorder;
pub mod sync;
pub mod venue;

use book::Delta;
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
    /// Messages that failed to parse (counted, then skipped).
    pub parse_errors: u64,
}

impl Handler {
    /// `window`: arbitration hold, in messages; `cap`: resync buffer size.
    pub fn new(venue: Venue, window: usize, cap: usize) -> Self {
        Self {
            venue,
            arb: Arbiter::new(window),
            engine: SyncEngine::new(cap),
            out: Vec::with_capacity(64),
            parse_errors: 0,
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
        let d = match self.parse(raw) {
            Ok(Msg::Delta(d)) => d,
            Ok(_) => return Action::None,
            Err(_) => {
                self.parse_errors += 1;
                return Action::None;
            }
        };
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
