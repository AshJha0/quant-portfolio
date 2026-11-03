//! Replay a QPRC recording through `qp_ull_feed` and print, after every
//! record, `index stale seq digest`: `seq` is the engine's last update id
//! (`-1` for Coinbase level2, which has none) and `digest` is an additive
//! hash over the book's levels, defined in tools/recorder/qp_recorder/refbook.py.
//!
//! Usage: `qp-recorder-replay binance|coinbase-level2 FILE SYMBOL`.
//! Accepts the C++ `QPRC` header (rewritten to the crate's `QPRR1` magic,
//! the records are identical) or a `QPRR1` file.

use std::io::Write;

use qp_ull_feed::book::{CryptoBook, Side};
use qp_ull_feed::recorder::{KIND_SNAPSHOT, MAGIC, Replay};
use qp_ull_feed::venue::{self, Msg};
use qp_ull_feed::{Handler, Venue};

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Sum over levels of `mix(mix(px) ^ qty ^ tag(side))`, mod 2^64; the same
/// digest as tools/recorder/qp_recorder/refbook.py and cpp/replay_dump.cpp.
fn digest(b: &CryptoBook) -> u64 {
    let mut h: u64 = 0;
    for (side, tag) in [(Side::Bid, 0u64), (Side::Ask, 0xa5a5a5a5a5a5a5a5)] {
        for (p, q) in b.levels(side) {
            if q > 0 {
                h = h.wrapping_add(mix(mix(p as u64) ^ q as u64 ^ tag));
            }
        }
    }
    h
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 4 {
        eprintln!("usage: qp-recorder-replay binance|coinbase-level2 FILE SYMBOL");
        std::process::exit(2);
    }
    let mut raw = std::fs::read(&a[2]).expect("read recording");
    if raw.starts_with(b"QPRC") {
        raw.splice(..8, MAGIC.iter().copied());
    }
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::new(out.lock());
    let frames = Replay::new(&raw[..]).unwrap_or_else(|_| {
        eprintln!("not a recording");
        std::process::exit(2)
    });
    match a[1].as_str() {
        "binance" => {
            // window 64: arbitration hold, as in the crate's own tests;
            // cap 100k: resync buffer.
            let mut h = Handler::new(Venue::BinanceDiffDepth, 64, 100_000);
            for (i, f) in frames.enumerate() {
                let Ok(f) = f else {
                    eprintln!("truncated final record");
                    break;
                };
                if f.kind == KIND_SNAPSHOT {
                    h.on_snapshot(&f.bytes);
                } else {
                    h.on_stream(usize::from(f.line.min(1)), &f.bytes);
                }
                writeln!(
                    out,
                    "{i} {} {} {:016x}",
                    u8::from(h.is_stale()),
                    h.engine.last_id(),
                    digest(h.engine.book())
                )
                .unwrap();
            }
        }
        "coinbase-level2" => {
            let product = &a[3];
            let mut book = CryptoBook::new();
            let mut live = false;
            for (i, f) in frames.enumerate() {
                let Ok(f) = f else {
                    eprintln!("truncated final record");
                    break;
                };
                let other = std::str::from_utf8(&f.bytes).is_ok_and(|s| {
                    s.contains("\"product_id\"")
                        && !s.contains(&format!("\"product_id\":\"{product}\""))
                });
                if !other {
                    match venue::coinbase_level2(&f.bytes) {
                        Ok(Msg::Snapshot(s)) => {
                            book.reset(&s);
                            live = true;
                        }
                        Ok(Msg::Unsequenced(ch)) if live => {
                            for c in &ch {
                                if book.apply(c).is_err() {
                                    live = false;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                writeln!(out, "{i} {} -1 {:016x}", u8::from(!live), digest(&book)).unwrap();
            }
        }
        v => {
            eprintln!("unknown venue {v}");
            std::process::exit(2);
        }
    }
}
