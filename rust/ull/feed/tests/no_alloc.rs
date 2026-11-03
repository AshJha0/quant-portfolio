//! W30 acceptance "no allocation on the hot path", for the crypto feed.
//!
//! A counting global allocator wraps the system one. After the handler is
//! built and synced from a snapshot, feeding a generated Binance day on both
//! lines (every message arrives twice: one copy wins, one is a duplicate)
//! must allocate zero times, and the book must still equal the reference.
//! A second test checks the fast decoder against the generic parser.

use qp_ull_feed::book::{Delta, Side};
use qp_ull_feed::fast::{Fast, binance_into};
use qp_ull_feed::generator::{Generator, Rng};
use qp_ull_feed::sync::Action;
use qp_ull_feed::venue::{Msg, binance};
use qp_ull_feed::{Handler, Venue};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct Counting;

thread_local! {
    // Per thread, because the test harness runs tests in parallel and the
    // other test allocates freely. A const-initialised Cell needs no heap.
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn allocs() -> u64 {
    ALLOCS.with(Cell::get)
}

fn bump() {
    // try_with: the allocator can be called during thread teardown.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

// SAFETY: forwards to System and only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        bump();
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

#[test]
fn binance_hot_path_allocates_nothing() {
    let mut g = Generator::new(5, 1000);
    let snap = g.snapshot_json();
    let events: Vec<Vec<u8>> = (0..50_000)
        .map(|_| g.next_event().raw.into_bytes())
        .collect();
    let (rb, ra) = (g.levels(Side::Bid), g.levels(Side::Ask));
    let mut h = Handler::new(Venue::BinanceDiffDepth, 8, 4096);
    assert_eq!(h.on_stream(0, &events[0]), Action::RequestSnapshot);
    // The snapshot is as of the event before events[0], so events[0] bridges.
    assert_eq!(h.on_snapshot(snap.as_bytes()), Action::None);
    assert!(!h.is_stale());
    // The counter sees this thread's allocations (self-check).
    let probe = allocs();
    drop(std::hint::black_box(vec![0u8; 16]));
    assert_eq!(allocs(), probe + 1);
    let before = allocs();
    for e in &events[1..] {
        h.on_stream(0, e);
        h.on_stream(1, e);
    }
    let after = allocs();
    assert_eq!(
        after - before,
        0,
        "hot path allocated {} times",
        after - before
    );
    assert!(!h.is_stale());
    assert_eq!(h.fast_decoded, 2 * (events.len() as u64 - 1) + 1);
    assert_eq!(h.arb_stats().duplicates, events.len() as u64 - 1);
    assert_eq!(h.engine.book().levels(Side::Bid), rb);
    assert_eq!(h.engine.book().levels(Side::Ask), ra);
}

fn fresh() -> Delta {
    Delta {
        first: 0,
        last: 0,
        changes: Vec::with_capacity(4096),
    }
}

#[test]
fn fast_decoder_agrees_with_generic_parser() {
    let mut g = Generator::new(9, 1);
    let mut r = Rng::new(77);
    let mut d = fresh();
    let (mut fast, mut fuzzed_fast, mut fuzzed) = (0u64, 0u64, 0u64);
    for i in 0..20_000 {
        let raw = g.next_event().raw;
        // Every generated event takes the fast path and matches.
        assert_eq!(binance_into(raw.as_bytes(), &mut d), Fast::Delta);
        assert_eq!(binance(raw.as_bytes()), Ok(Msg::Delta(d.clone())));
        fast += 1;
        // Wrapped in the combined-stream envelope, too.
        if i % 10 == 0 {
            let w = format!(r#"{{"stream":"qpusdt@depth","data":{raw}}}"#);
            assert_eq!(binance_into(w.as_bytes(), &mut d), Fast::Delta);
            assert_eq!(binance(w.as_bytes()), Ok(Msg::Delta(d.clone())));
        }
        // Mutated bytes: whenever the fast path accepts, it must agree.
        let mut m = raw.into_bytes();
        match r.below(3) {
            0 => {
                let k = r.below(m.len() as u64) as usize;
                m.truncate(k);
            }
            1 => {
                for _ in 0..1 + r.below(3) {
                    let k = r.below(m.len() as u64) as usize;
                    m[k] = r.next_u64() as u8;
                }
            }
            _ => {
                let k = r.below(m.len() as u64) as usize;
                let c = b" ,:[]{}\"0.9e-\\"[r.below(14) as usize];
                m.insert(k, c);
            }
        }
        fuzzed += 1;
        if binance_into(&m, &mut d) == Fast::Delta {
            fuzzed_fast += 1;
            assert_eq!(
                binance(&m),
                Ok(Msg::Delta(d.clone())),
                "{}",
                String::from_utf8_lossy(&m)
            );
        } else {
            assert!(d.changes.is_empty());
        }
    }
    assert_eq!(fast, 20_000);
    // Some mutations stay valid (a digit changed inside a price, say).
    assert!(
        fuzzed_fast > 0 && fuzzed_fast < fuzzed,
        "{fuzzed_fast}/{fuzzed}"
    );
}

#[test]
fn window_move_marks_stale_and_recovers() {
    use qp_ull_feed::book::ProductSpec;
    // A narrow window (64 ticks of 0.01 around the touch) on a generated day:
    // the mid drifts, so the touch leaves the window, the book goes stale,
    // and a fresh snapshot re-centres it. Whenever the handler is fresh its
    // whole book (flat window plus side store) equals the reference.
    let spec = ProductSpec {
        n_levels: 64,
        ..ProductSpec::default()
    };
    let mut g = Generator::new(3, 1);
    let mut h = Handler::with_spec(Venue::BinanceDiffDepth, 4, 4096, spec);
    let (mut recoveries, mut checked) = (0, 0);
    for _ in 0..20_000 {
        let e = g.next_event();
        if h.on_stream(0, e.raw.as_bytes()) == Action::RequestSnapshot {
            // Answer at once with the book as of this event.
            if h.on_snapshot(g.snapshot_json().as_bytes()) == Action::None {
                recoveries += 1;
            }
        }
        if !h.is_stale() {
            let (rb, ra) = (g.levels(Side::Bid), g.levels(Side::Ask));
            // Deep levels sit in the side store: the whole book is exact.
            assert_eq!(h.engine.book().levels(Side::Bid), rb);
            assert_eq!(h.engine.book().levels(Side::Ask), ra);
            checked += 1;
        }
    }
    assert!(h.engine.stats.out_of_window > 0);
    assert!(recoveries > 1, "{recoveries}");
    assert!(checked > 10_000, "{checked}");
}

#[test]
fn order_feed_book_ops_allocate_nothing() {
    use qp_ull_feed::book::{Change, CryptoBook, Snapshot};
    let mut b = CryptoBook::new();
    let s = 100_000_000i64; // 1.0
    b.reset(&Snapshot {
        last: 1,
        levels: vec![],
        orders: vec![(1, Side::Bid, 100 * s, s), (2, Side::Ask, 101 * s, s)],
    })
    .unwrap();
    let mut r = Rng::new(1);
    let mut live: Vec<u128> = Vec::with_capacity(1024);
    let mut next: u128 = 1 << 100; // UUID-sized ids
    let before = allocs();
    for _ in 0..100_000 {
        let c = if live.len() < 500 && (live.is_empty() || r.below(2) == 0) {
            next += 0x1_0000_0001;
            live.push(next);
            let side = if r.below(2) == 0 {
                Side::Bid
            } else {
                Side::Ask
            };
            let px =
                if side == Side::Bid { 99 * s } else { 102 * s } - (r.below(50) as i64) * s / 100;
            Change::Open {
                id: next,
                side,
                px,
                qty: (1 + r.below(9) as i64) * s / 10,
            }
        } else {
            let k = r.below(live.len() as u64) as usize;
            let id = live.swap_remove(k);
            Change::Done { id }
        };
        b.apply(&c).unwrap();
    }
    assert_eq!(allocs() - before, 0);
    assert_eq!(b.order_count(), live.len() + 2);
    assert!(b.check_invariant());
}
