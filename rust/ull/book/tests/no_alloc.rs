//! W30 acceptance "no allocation on the hot path": a counting global
//! allocator wraps the system one; after the book is built and the day is
//! decoded into memory, applying every message must allocate zero times.

use qp_ull_book::Book;
use qp_ull_qpbf::Packet;
use qp_ull_qpbf::generator::{GenConfig, generate_day};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards to System and only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

#[test]
fn decode_and_apply_allocate_nothing() {
    let cfg = GenConfig {
        seed: 11,
        instruments: 2,
        max_live: 500,
        ..GenConfig::default()
    };
    let mid = cfg.start_mid;
    let mut bytes: Vec<Vec<u8>> = Vec::new();
    generate_day(cfg, 1, 200_000, |p| bytes.push(p.to_vec()));
    let mut books = [
        Book::new(mid - 30_000, 60_000, 2_048),
        Book::new(mid - 30_000, 60_000, 2_048),
    ];
    let before = ALLOCS.load(Ordering::SeqCst);
    let mut n = 0u64;
    for pkt in &bytes {
        for (_, m) in Packet::parse(pkt).unwrap().messages() {
            if m.instrument() != 0 {
                books[(m.instrument() - 1) as usize].apply(&m).unwrap();
            }
            n += 1;
        }
    }
    let after = ALLOCS.load(Ordering::SeqCst);
    assert_eq!(n, 200_000);
    assert_eq!(
        after - before,
        0,
        "hot path allocated {} times",
        after - before
    );
}
