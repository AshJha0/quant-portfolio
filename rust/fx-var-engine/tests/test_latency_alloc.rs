//! Zero heap allocations on the success paths of the hot kernels.
//!
//! A counting global allocator wraps the system one. This file is its own
//! test binary with a single `#[test]`, so nothing else allocates while a
//! kernel is measured.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use fx_var_engine::prelude::*;

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

/// Heap allocations made while running `f`.
fn allocs<R>(f: impl FnOnce() -> R) -> (usize, R) {
    let before = ALLOCS.load(Ordering::SeqCst);
    let r = std::hint::black_box(f());
    (ALLOCS.load(Ordering::SeqCst) - before, r)
}

#[test]
fn hot_kernels_do_not_allocate() {
    let market = Market::new(
        [("EUR", 1.10), ("JPY", 0.0090), ("GBP", 1.27)],
        [("USD", 0.05), ("EUR", 0.03), ("JPY", 0.001), ("GBP", 0.04)],
    )
    .unwrap();
    let book = Book::new(
        vec![
            Position::Spot(SpotPosition::new("EURUSD", 10e6, None)),
            Position::Forward(ForwardPosition::new("USDJPY", 5e6, 0.5, None)),
            Position::Spot(SpotPosition::new("GBPJPY", -3e6, None)),
        ],
        "EUR",
    )
    .unwrap();
    let cb = CompiledBook::new(&book, &market).unwrap();
    let k = cb.factors().len();
    let shocks: Vec<f64> = (0..k).map(|j| 0.003 * (j as f64 - 1.0)).collect();
    let mut scen = Matrix::zeros(64, k);
    for s in 0..64 {
        for j in 0..k {
            scen.set(s, j, 0.004 * ((s * 7 + j * 3) as f64).sin());
        }
    }
    let mut out = vec![0.0; 64];
    let pnl: Vec<f64> = (0..1000)
        .map(|i| 1000.0 * (0.37 * i as f64).sin())
        .collect();
    let mut ws = vec![0.0; pnl.len()];
    let cov = Matrix::identity(k);
    let w = vec![1.0e6; k];

    let (n, r) = allocs(|| cb.pnl_row(&shocks));
    assert!(r.is_ok());
    assert_eq!(n, 0, "CompiledBook::pnl_row allocated");

    let (n, r) = allocs(|| cb.pnl(&shocks));
    assert!(r.is_ok());
    assert_eq!(n, 0, "CompiledBook::pnl allocated");

    let (n, r) = allocs(|| cb.pnl_aligned_into(&scen, &mut out));
    assert!(r.is_ok());
    assert_eq!(n, 0, "CompiledBook::pnl_aligned_into allocated");

    let (n, r) = allocs(|| empirical_var_es_into(&pnl, 0.01, &mut ws));
    assert!(r.is_ok());
    assert_eq!(n, 0, "empirical_var_es_into allocated");

    let (n, r) = allocs(|| empirical_var_into(&pnl, 0.01, &mut ws));
    assert!(r.is_ok());
    assert_eq!(n, 0, "empirical_var_into allocated");

    let (n, r) = allocs(|| portfolio_sigma(&w, &cov));
    assert!(r.is_ok());
    assert_eq!(n, 0, "portfolio_sigma allocated");

    // The error path of the new kernels does not allocate either.
    let (n, r) = allocs(|| empirical_var_es_into(&pnl, 0.7, &mut ws));
    assert_eq!(r, Err(KernelError::BadAlpha));
    assert_eq!(n, 0, "KernelError construction allocated");
}
