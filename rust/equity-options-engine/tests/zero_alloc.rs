//! Heap-allocation counter for the kernel success paths.
//!
//! This file is its own test binary with a counting global allocator and
//! a single `#[test]`, so nothing else runs while a call is measured.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use eq_options_engine::fast::{
    bs_greeks_batch, bs_price_batch, try_black76_greeks, try_black76_price, try_bs_greeks,
    try_bs_price, try_crr_price, try_implied_vol, try_mc_price, BsBatch, GreeksOut,
};
use eq_options_engine::{
    black76_greeks, black76_price, bs_greeks, bs_price, crr_price, implied_vol, CrrWorkspace,
    Exercise, McWorkspace, OptionType,
};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap allocations made by `f`.
fn allocs<R>(f: impl FnOnce() -> R) -> usize {
    let before = ALLOCS.load(Ordering::SeqCst);
    let r = f();
    let after = ALLOCS.load(Ordering::SeqCst);
    std::hint::black_box(r);
    after - before
}

#[test]
fn kernel_success_paths_do_not_allocate() {
    let (s, k, t, r, v, q) = (100.0, 105.0, 0.75, 0.03, 0.25, 0.01);
    let call = OptionType::Call;
    let put = OptionType::Put;

    // The counter is live: the classic tree builds its buffers per call.
    assert!(allocs(|| crr_price(s, k, t, r, v, q, put, Exercise::American, 200)) > 0);

    // Closed-form kernels, classic and fast.
    assert_eq!(allocs(|| bs_price(s, k, t, r, v, q, call).unwrap()), 0);
    assert_eq!(allocs(|| bs_greeks(s, k, t, r, v, q, put).unwrap()), 0);
    assert_eq!(allocs(|| black76_price(s, k, t, r, v, call).unwrap()), 0);
    assert_eq!(allocs(|| black76_greeks(s, k, t, r, v, put).unwrap()), 0);
    assert_eq!(allocs(|| try_bs_price(s, k, t, r, v, q, call).unwrap()), 0);
    assert_eq!(allocs(|| try_bs_greeks(s, k, t, r, v, q, put).unwrap()), 0);
    assert_eq!(
        allocs(|| try_black76_price(s, k, t, r, v, call).unwrap()),
        0
    );
    assert_eq!(
        allocs(|| try_black76_greeks(s, k, t, r, v, put).unwrap()),
        0
    );
    let p = bs_price(s, k, t, r, v, q, call).unwrap();
    assert_eq!(allocs(|| implied_vol(p, s, k, t, r, q, call).unwrap()), 0);
    assert_eq!(
        allocs(|| try_implied_vol(p, s, k, t, r, q, call).unwrap()),
        0
    );

    // Error paths of the fast API do not allocate either.
    assert_eq!(
        allocs(|| try_bs_price(f64::NAN, k, t, r, v, q, call).is_err()),
        0
    );
    assert_eq!(
        allocs(|| try_implied_vol(0.0, s, k, t, r, q, call).is_err()),
        0
    );

    // Tree: a pre-sized workspace never grows.
    let mut ws = CrrWorkspace::with_steps(1000);
    for (ex, n) in [
        (Exercise::American, 1000),
        (Exercise::European, 1000),
        (Exercise::American, 10),
    ] {
        assert_eq!(
            allocs(|| try_crr_price(s, k, t, r, v, q, put, ex, n, &mut ws).unwrap()),
            0,
            "{ex:?} n={n}"
        );
    }

    // Monte Carlo, single thread: after one warm-up call sizes the
    // workspace, repeated calls do not allocate.
    let mut mc = McWorkspace::new();
    for (anti, cv) in [(true, true), (false, true), (true, false), (false, false)] {
        try_mc_price(s, k, t, r, v, q, call, 20_000, anti, cv, 1, 1, &mut mc).unwrap();
        assert_eq!(
            allocs(
                || try_mc_price(s, k, t, r, v, q, call, 20_000, anti, cv, 2, 1, &mut mc).unwrap()
            ),
            0,
            "anti={anti} cv={cv}"
        );
    }

    // Batch pricers write into caller slices.
    let sv = [90.0, 100.0, 110.0, 120.0];
    let kv = [100.0; 4];
    let tv = [0.5; 4];
    let rv = [0.03; 4];
    let vv = [0.2; 4];
    let qv = [0.01; 4];
    let ot = [call, put, call, put];
    let b = BsBatch {
        s: &sv,
        k: &kv,
        t: &tv,
        r: &rv,
        sigma: &vv,
        q: &qv,
        option_type: &ot,
    };
    let mut out = [0.0; 4];
    assert_eq!(allocs(|| bs_price_batch(&b, &mut out).unwrap()), 0);
    let mut g = [[0.0; 4]; 8];
    let [g0, g1, g2, g3, g4, g5, g6, g7] = &mut g;
    let mut go = GreeksOut {
        price: g0,
        delta: g1,
        gamma: g2,
        vega: g3,
        theta: g4,
        rho: g5,
        vanna: g6,
        volga: g7,
    };
    assert_eq!(allocs(|| bs_greeks_batch(&b, &mut go).unwrap()), 0);
}
