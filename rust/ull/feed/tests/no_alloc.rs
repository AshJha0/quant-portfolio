//! W30 acceptance "no allocation on the hot path", for the crypto feed.
//!
//! A counting global allocator wraps the system one. After the handler is
//! built and synced from a snapshot, feeding a generated Binance day on both
//! lines (every message arrives twice: one copy wins, one is a duplicate)
//! must allocate zero times, and the book must still equal the reference.
//! A second test checks the fast decoder against the generic parser. The
//! Coinbase level2 decoder gets the same two checks, on generated messages,
//! mutated bytes and the live recordings in tools/recorder/recordings.

use qp_ull_feed::book::{Change, Delta, Side};
use qp_ull_feed::fast::{Fast, binance_into, coinbase_level2_into};
use qp_ull_feed::generator::{Generator, Rng};
use qp_ull_feed::sync::Action;
use qp_ull_feed::venue::{Msg, binance, coinbase_level2};
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

// ---- Coinbase level2 --------------------------------------------------------

/// Bit reader and canonical Huffman decoding for RFC 1951. The crate is std
/// only, and the recordings are gzip files, so the test carries an inflater.
struct Bits<'a> {
    b: &'a [u8],
    i: usize,
    buf: u32,
    n: u32,
}

impl Bits<'_> {
    fn take(&mut self, k: u32) -> u32 {
        while self.n < k {
            self.buf |= u32::from(self.b[self.i]) << self.n;
            self.i += 1;
            self.n += 8;
        }
        let v = self.buf & ((1u32 << k) - 1);
        self.buf >>= k;
        self.n -= k;
        v
    }
    fn sym(&mut self, h: &Huff) -> usize {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= self.take(1) as i32;
            let c = h.count[len];
            if code - c < first {
                return h.sym[(index + code - first) as usize];
            }
            index += c;
            first = (first + c) << 1;
            code <<= 1;
        }
        panic!("bad Huffman code");
    }
}

struct Huff {
    count: [i32; 16],
    sym: Vec<usize>,
}

fn huff(lens: &[u8]) -> Huff {
    let mut count = [0i32; 16];
    for &l in lens {
        count[l as usize] += 1;
    }
    count[0] = 0;
    let mut sym = Vec::with_capacity(lens.len());
    for len in 1..16u8 {
        sym.extend((0..lens.len()).filter(|&s| lens[s] == len));
    }
    Huff { count, sym }
}

fn inflate_block(r: &mut Bits<'_>, lit: &Huff, dist: &Huff, out: &mut Vec<u8>) {
    const LEN: [usize; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEXT: [u32; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST: [usize; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DEXT: [u32; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];
    loop {
        let s = r.sym(lit);
        match s {
            0..=255 => out.push(s as u8),
            256 => return,
            _ => {
                let len = LEN[s - 257] + r.take(LEXT[s - 257]) as usize;
                let d = r.sym(dist);
                let back = DIST[d] + r.take(DEXT[d]) as usize;
                for _ in 0..len {
                    out.push(out[out.len() - back]);
                }
            }
        }
    }
}

fn gunzip(b: &[u8]) -> Vec<u8> {
    assert_eq!(&b[..3], &[0x1f, 0x8b, 8]);
    let flags = b[3];
    let mut i = 10;
    if flags & 4 != 0 {
        i += 2 + usize::from(b[i]) + (usize::from(b[i + 1]) << 8);
    }
    for bit in [8, 16] {
        if flags & bit != 0 {
            while b[i] != 0 {
                i += 1;
            }
            i += 1;
        }
    }
    if flags & 2 != 0 {
        i += 2;
    }
    let mut r = Bits { b, i, buf: 0, n: 0 };
    let mut out = Vec::new();
    loop {
        let last = r.take(1) == 1;
        match r.take(2) {
            0 => {
                (r.buf, r.n) = (0, 0);
                let n = usize::from(b[r.i]) | (usize::from(b[r.i + 1]) << 8);
                r.i += 4;
                out.extend_from_slice(&b[r.i..r.i + n]);
                r.i += n;
            }
            1 => {
                let mut l = [8u8; 288];
                l[144..256].fill(9);
                l[256..280].fill(7);
                inflate_block(&mut r, &huff(&l), &huff(&[5u8; 30]), &mut out);
            }
            2 => {
                const ORDER: [usize; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let nlen = r.take(5) as usize + 257;
                let ndist = r.take(5) as usize + 1;
                let ncode = r.take(4) as usize + 4;
                let mut cl = [0u8; 19];
                for &o in &ORDER[..ncode] {
                    cl[o] = r.take(3) as u8;
                }
                let h = huff(&cl);
                let mut l: Vec<u8> = Vec::with_capacity(nlen + ndist);
                while l.len() < nlen + ndist {
                    match r.sym(&h) {
                        s @ 0..=15 => l.push(s as u8),
                        16 => {
                            let p = *l.last().unwrap();
                            let n = 3 + r.take(2) as usize;
                            l.extend(std::iter::repeat_n(p, n));
                        }
                        17 => {
                            let n = 3 + r.take(3) as usize;
                            l.extend(std::iter::repeat_n(0, n));
                        }
                        _ => {
                            let n = 11 + r.take(7) as usize;
                            l.extend(std::iter::repeat_n(0, n));
                        }
                    }
                }
                inflate_block(&mut r, &huff(&l[..nlen]), &huff(&l[nlen..]), &mut out);
            }
            _ => panic!("bad deflate block"),
        }
        if last {
            break;
        }
    }
    let t = &b[b.len() - 4..];
    assert_eq!(
        u32::from_le_bytes([t[0], t[1], t[2], t[3]]),
        out.len() as u32
    );
    out
}

/// Every record of every recorded Coinbase session.
fn coinbase_frames() -> Vec<Vec<u8>> {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../tools/recorder/recordings"
    );
    let mut frames = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if !(name.starts_with("coinbase_") && name.ends_with(".qprc.gz")) {
            continue;
        }
        let raw = gunzip(&std::fs::read(&p).unwrap());
        assert_eq!(&raw[..4], b"QPRC");
        let mut i = 8;
        while i + 14 <= raw.len() {
            let n = u32::from_le_bytes(raw[i + 10..i + 14].try_into().unwrap()) as usize;
            frames.push(raw[i + 14..i + 14 + n].to_vec());
            i += 14 + n;
        }
    }
    assert!(!frames.is_empty(), "no Coinbase recording in {dir}");
    frames
}

fn is_l2update(f: &[u8]) -> bool {
    matches!(coinbase_level2(f), Ok(Msg::Unsequenced(_)))
}

fn generated_l2update(r: &mut Rng) -> String {
    let n = r.below(6);
    let mut c = String::new();
    for k in 0..n {
        if k > 0 {
            c.push(',');
        }
        let side = if r.below(2) == 0 { "buy" } else { "sell" };
        let (p, f, q) = (70_000 + r.below(9_000), r.below(100), r.below(100_000_000));
        c += &format!(r#"["{side}","{p}.{f:02}","0.{q:08}"]"#);
    }
    let (t, ch) = (r#""type":"l2update""#, format!(r#""changes":[{c}]"#));
    let (pid, time) = (
        r#""product_id":"BTC-USD""#,
        r#""time":"2026-11-03T19:08:00.498600Z""#,
    );
    match r.below(4) {
        0 => format!("{{{ch},{t},{time},{pid}}}"),
        1 => format!(" {{ {t} , {pid} ,\n{ch} , {time} }}\r\n"),
        _ => format!("{{{t},{pid},{ch},{time}}}"),
    }
}

/// Decode `raw`; when the fast path accepts, the generic parser must agree.
fn agrees(raw: &[u8], out: &mut Vec<Change>) -> bool {
    if coinbase_level2_into(raw, out) == Fast::Delta {
        assert_eq!(
            coinbase_level2(raw),
            Ok(Msg::Unsequenced(out.clone())),
            "{}",
            String::from_utf8_lossy(raw)
        );
        true
    } else {
        assert!(out.is_empty());
        false
    }
}

fn mutate(r: &mut Rng, m: &mut Vec<u8>) {
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
}

#[test]
fn coinbase_level2_decoder_allocates_nothing() {
    let frames = coinbase_frames();
    let updates = frames.iter().filter(|f| is_l2update(f)).count() as u64;
    let mut out: Vec<Change> = Vec::with_capacity(4096);
    let before = allocs();
    let mut fast = 0u64;
    for _ in 0..20 {
        for f in &frames {
            if coinbase_level2_into(f, &mut out) == Fast::Delta {
                fast += 1;
            }
        }
    }
    let after = allocs();
    assert_eq!(after - before, 0, "decoder allocated {}", after - before);
    assert!(updates > 0);
    assert_eq!(fast, 20 * updates);
    // Contrast: the generic parser allocates on every frame.
    let g0 = allocs();
    for f in &frames {
        std::hint::black_box(coinbase_level2(f)).ok();
    }
    assert!(allocs() - g0 > frames.len() as u64);
}

#[test]
fn coinbase_level2_decoder_agrees_with_generic_parser() {
    let mut r = Rng::new(78);
    let mut out: Vec<Change> = Vec::with_capacity(4096);
    let (mut fuzzed, mut fuzzed_fast) = (0u64, 0u64);
    // Live frames: every l2update is accepted and matches.
    let frames = coinbase_frames();
    for f in &frames {
        assert_eq!(agrees(f, &mut out), is_l2update(f));
        for _ in 0..50 {
            let mut m = f.clone();
            mutate(&mut r, &mut m);
            fuzzed += 1;
            fuzzed_fast += u64::from(agrees(&m, &mut out));
        }
    }
    for _ in 0..20_000 {
        let mut m = generated_l2update(&mut r).into_bytes();
        assert!(agrees(&m, &mut out));
        mutate(&mut r, &mut m);
        fuzzed += 1;
        fuzzed_fast += u64::from(agrees(&m, &mut out));
    }
    println!("fuzzed {fuzzed}, accepted {fuzzed_fast}");
    assert!(
        fuzzed_fast > 0 && fuzzed_fast < fuzzed,
        "{fuzzed_fast}/{fuzzed}"
    );
}
