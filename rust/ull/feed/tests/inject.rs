//! Gap-injection acceptance: drop, duplicate, reorder and burst on lines A
//! and B. Invariant after every message: when the handler is not stale, its
//! book equals the reference book at the handler's last id. Every pattern
//! must converge to the reference at the end.

use std::collections::HashMap;

use qp_ull_feed::book::Side;
use qp_ull_feed::generator::{Generator, Rng, fmt_fixed};
use qp_ull_feed::recorder::{KIND_SNAPSHOT, KIND_STREAM, Recorder, Replay};
use qp_ull_feed::sync::Action;
use qp_ull_feed::{Handler, Venue};

type Lv = Vec<(i64, i64)>;

struct Day {
    events: Vec<String>,
    refs: HashMap<u64, (Lv, Lv)>,
    snaps: Vec<(u64, String)>, // (last id, Binance snapshot json) after each event
}

fn snap_json(id: u64, b: &Lv, a: &Lv) -> String {
    let f = |v: &Lv| {
        v.iter()
            .map(|(p, q)| format!("[\"{}\",\"{}\"]", fmt_fixed(*p), fmt_fixed(*q)))
            .collect::<Vec<_>>()
            .join(",")
    };
    format!(
        "{{\"lastUpdateId\":{id},\"bids\":[{}],\"asks\":[{}]}}",
        f(b),
        f(a)
    )
}

fn day(seed: u64, n: usize) -> Day {
    let mut g = Generator::new(seed, 1000);
    let mut d = Day {
        events: vec![],
        refs: HashMap::new(),
        snaps: vec![],
    };
    let s0 = (g.levels(Side::Bid), g.levels(Side::Ask));
    d.refs.insert(g.last_id(), s0);
    for _ in 0..n {
        let e = g.next_event();
        let (b, a) = (g.levels(Side::Bid), g.levels(Side::Ask));
        d.snaps.push((e.last, snap_json(e.last, &b, &a)));
        d.refs.insert(e.last, (b, a));
        d.events.push(e.raw);
    }
    d
}

/// Run lines A and B. Position i of each list is time slot i (A then B);
/// `None` is a slot where that line lost its packet.
/// A snapshot request is answered `delay` messages later with the book as of
/// the newest event delivered so far. Returns (handler, ever_stale_after_sync, recording).
fn run(d: &Day, a: &[Option<usize>], b: &[Option<usize>], delay: usize) -> (Handler, u64, Vec<u8>) {
    let mut h = Handler::new(Venue::BinanceDiffDepth, 8, 4096);
    let mut rec = Recorder::new(Vec::new()).unwrap();
    let mut pending: Option<usize> = None; // countdown
    let mut newest = 0usize;
    let mut stale_msgs_after_first_sync = 0u64;
    let mut synced_once = false;
    let n = a.len().max(b.len());
    let mut t = 0u64;
    for i in 0..n {
        for (line, l) in [(0usize, a), (1, b)] {
            let Some(&Some(ix)) = l.get(i) else { continue };
            newest = newest.max(ix);
            t += 1;
            rec.record(t, line as u8, KIND_STREAM, d.events[ix].as_bytes())
                .unwrap();
            if h.on_stream(line, d.events[ix].as_bytes()) == Action::RequestSnapshot {
                assert!(pending.is_none(), "duplicate snapshot request");
                pending = Some(delay);
            }
            if let Some(c) = pending {
                if c == 0 {
                    pending = None;
                    let s = d.snaps[newest].1.as_bytes();
                    t += 1;
                    rec.record(t, line as u8, KIND_SNAPSHOT, s).unwrap();
                    if h.on_snapshot(s) == Action::RequestSnapshot {
                        pending = Some(delay);
                    }
                } else {
                    pending = Some(c - 1);
                }
            }
            if h.is_stale() {
                if synced_once {
                    stale_msgs_after_first_sync += 1;
                }
            } else {
                synced_once = true;
                let (rb, ra) = &d.refs[&h.engine.last_id()];
                assert_eq!(
                    &h.engine.book().levels(Side::Bid),
                    rb,
                    "bids diverged at id {}",
                    h.engine.last_id()
                );
                assert_eq!(
                    &h.engine.book().levels(Side::Ask),
                    ra,
                    "asks diverged at id {}",
                    h.engine.last_id()
                );
            }
        }
    }
    // Drain: answer an outstanding request at end of day.
    if pending.is_some() {
        let s = d.snaps[newest].1.as_bytes();
        rec.record(t + 1, 0, KIND_SNAPSHOT, s).unwrap();
        h.on_snapshot(s);
    }
    (h, stale_msgs_after_first_sync, rec.finish().unwrap())
}

fn on(v: &[usize]) -> Vec<Option<usize>> {
    v.iter().map(|&i| Some(i)).collect()
}

fn converged(d: &Day, h: &Handler) {
    assert!(!h.is_stale(), "did not converge");
    let last = d.snaps.last().unwrap().0;
    assert_eq!(h.engine.last_id(), last);
    let (rb, ra) = &d.refs[&last];
    assert_eq!(&h.engine.book().levels(Side::Bid), rb);
    assert_eq!(&h.engine.book().levels(Side::Ask), ra);
}

const N: usize = 3000;

#[test]
fn clean_ab() {
    let d = day(1, N);
    let all: Vec<usize> = (0..N).collect();
    let (h, stale, _) = run(&d, &on(&all), &on(&all), 3);
    converged(&d, &h);
    assert_eq!(stale, 0);
    assert_eq!(h.engine.stats.gaps, 0);
    assert_eq!(h.arb_stats().duplicates as usize, N); // every B copy
}

#[test]
fn drop_on_one_line_is_covered_by_the_other() {
    let d = day(2, N);
    let mut r = Rng::new(7);
    let mut a = vec![];
    let mut b = vec![];
    for i in 0..N {
        // Each line loses about 10%; never both in the same slot.
        match r.below(10) {
            0 => (a.push(None), b.push(Some(i))),
            1 => (a.push(Some(i)), b.push(None)),
            _ => (a.push(Some(i)), b.push(Some(i))),
        };
    }
    let (h, stale, _) = run(&d, &a, &b, 3);
    converged(&d, &h);
    assert_eq!(
        h.engine.stats.gaps, 0,
        "a single-line drop must not reach the engine"
    );
    assert_eq!(stale, 0);
}

#[test]
fn drop_on_both_lines_goes_stale_and_resyncs() {
    let d = day(3, N);
    let lost = [500usize, 1500, 2500];
    let l: Vec<usize> = (0..N).filter(|i| !lost.contains(i)).collect();
    let (h, stale, _) = run(&d, &on(&l), &on(&l), 5);
    converged(&d, &h);
    assert_eq!(h.engine.stats.gaps, 3);
    assert!(stale > 0, "book must be flagged stale during the gap");
    assert!(h.engine.stats.snapshots >= 4);
}

#[test]
fn duplicates_only() {
    let d = day(4, N);
    let a: Vec<usize> = (0..N).flat_map(|i| [i, i]).collect();
    let (h, stale, _) = run(&d, &on(&a), &[], 3);
    converged(&d, &h);
    assert_eq!((stale, h.engine.stats.gaps), (0, 0));
}

#[test]
fn reorder_within_window_is_absorbed() {
    let d = day(5, N);
    let mut a: Vec<usize> = (0..N).collect();
    let mut r = Rng::new(9);
    let mut i = 10;
    while i + 3 < N {
        if r.below(5) == 0 {
            a.swap(i, i + 2);
        }
        i += 4;
    }
    let (h, stale, _) = run(&d, &on(&a), &[], 3);
    converged(&d, &h);
    assert_eq!((stale, h.engine.stats.gaps), (0, 0));
}

#[test]
fn burst_loss_converges() {
    let d = day(6, N);
    let l: Vec<usize> = (0..N).filter(|i| !(1000..1050).contains(i)).collect();
    let (h, stale, _) = run(&d, &on(&l), &on(&l), 20);
    converged(&d, &h);
    assert!(stale >= 1);
    assert_eq!(h.engine.stats.gaps, 1);
}

#[test]
fn reorder_beyond_window_resyncs() {
    let d = day(8, N);
    let mut a: Vec<usize> = (0..N).collect();
    let x = a.remove(700);
    a.insert(740, x); // 40 messages late, window is 8
    let (h, stale, _) = run(&d, &on(&a), &[], 3);
    converged(&d, &h);
    assert!(stale > 0);
    assert!(h.engine.stats.gaps >= 1);
}

#[test]
fn recorded_day_replays_to_the_same_book() {
    let d = day(10, N);
    let l: Vec<usize> = (0..N).filter(|i| *i != 1234).collect();
    let (h, _, bytes) = run(&d, &on(&l), &on(&l), 4);
    let mut r = Handler::new(Venue::BinanceDiffDepth, 8, 4096);
    for f in Replay::new(&bytes[..]).unwrap() {
        let f = f.unwrap();
        if f.kind == KIND_STREAM {
            r.on_stream(f.line as usize, &f.bytes);
        } else {
            r.on_snapshot(&f.bytes);
        }
    }
    assert_eq!(r.engine.book(), h.engine.book());
    assert_eq!(r.engine.stats, h.engine.stats);
    assert_eq!(r.arb_stats(), h.arb_stats());
}

#[test]
fn fuzzed_messages_never_panic() {
    let d = day(11, 200);
    let mut r = Rng::new(42);
    let mut h = Handler::new(Venue::BinanceDiffDepth, 8, 64);
    let mut c = Handler::new(Venue::CoinbaseFull, 8, 64);
    for i in 0..20_000 {
        let mut m = d.events[i % d.events.len()].clone().into_bytes();
        match r.below(4) {
            0 => {
                let k = r.below(m.len() as u64) as usize;
                m.truncate(k);
            }
            1 => {
                for _ in 0..1 + r.below(4) {
                    let k = r.below(m.len() as u64) as usize;
                    m[k] = r.next_u64() as u8;
                }
            }
            2 => m = (0..r.below(64)).map(|_| r.next_u64() as u8).collect(),
            _ => {}
        }
        h.on_stream(0, &m);
        h.on_snapshot(&m);
        c.on_stream(1, &m);
        c.on_snapshot(&m);
    }
    assert!(h.parse_errors > 0);
}

#[test]
fn coinbase_full_gap_and_resync() {
    let id = |n: u32| format!("{n:08x}-0000-0000-0000-000000000000");
    let open = |s: u64, n: u32, px: &str| {
        format!(
            r#"{{"type":"open","sequence":{s},"order_id":"{}","side":"buy","price":"{px}","remaining_size":"1.5"}}"#,
            id(n)
        )
    };
    let mut h = Handler::new(Venue::CoinbaseFull, 4, 64);
    assert_eq!(
        h.on_stream(0, open(11, 1, "100.00").as_bytes()),
        Action::RequestSnapshot
    );
    h.on_stream(0, open(12, 2, "99.99").as_bytes());
    let snap = format!(
        r#"{{"sequence":10,"bids":[["100.01","2","{}"]],"asks":[]}}"#,
        id(9)
    );
    assert_eq!(h.on_snapshot(snap.as_bytes()), Action::None);
    assert!(!h.is_stale());
    assert_eq!(h.engine.book().order_count(), 3);
    let m = format!(
        r#"{{"type":"match","sequence":13,"maker_order_id":"{}","size":"0.5","side":"buy"}}"#,
        id(9)
    );
    h.on_stream(0, m.as_bytes());
    assert_eq!(
        h.engine.book().best_bid(),
        Some((10_001_000_000, 150_000_000))
    );
    assert!(h.engine.book().check_invariant());
    // Sequence 14 lost on the only line: after the window the gap reaches the engine.
    for s in 15..=20 {
        h.on_stream(
            0,
            format!(
                r#"{{"type":"received","sequence":{s},"order_id":"{}"}}"#,
                id(50)
            )
            .as_bytes(),
        );
    }
    assert!(h.is_stale());
}
