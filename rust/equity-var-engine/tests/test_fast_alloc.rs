//! Zero heap allocations on the success paths of `eq_var_engine::fast`.
//!
//! This file is its own test binary with one `#[test]`. The counter is
//! per thread, so it sees every allocation the measured call makes (the
//! fast kernels run on the calling thread) and nothing the test harness's
//! own threads do at the same time. A process-wide counter picked up stray
//! harness allocations on a CI runner. The counter is read just before and
//! just after each measured call.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use eq_var_engine::fast::*;
use eq_var_engine::prelude::{Matrix, Rng, TailModel};

struct Counting;

thread_local! {
    // Const-initialised and without a destructor, so touching it from
    // inside the allocator never allocates or registers a TLS destructor.
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn bump() {
    // `try_with` so an allocation during thread teardown cannot panic.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

fn allocs() -> usize {
    ALLOCS.with(Cell::get)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn count<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = allocs();
    let r = std::hint::black_box(f());
    (r, allocs() - before)
}

#[test]
fn fast_kernels_do_not_allocate() {
    let mut rng = Rng::new(11);
    let pnl: Vec<f64> = (0..500).map(|_| 1.0e4 * rng.standard_normal()).collect();
    let mut work = vec![0.0; pnl.len()];
    let mut est = vec![0.0; 64];
    let n = 5;
    let mut cov = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            cov.set(i, j, if i == j { 4e-4 } else { 1e-4 });
        }
    }
    let w = vec![1.0e6, -2.0e5, 3.0e5, 4.0e5, -1.0e5];
    let t = TailModel::StudentT { df: 5.0 };

    let (r, a) = count(|| historical_var_ws(&pnl, 0.01, &mut work));
    assert!(r.is_ok());
    assert_eq!(a, 0, "historical_var_ws allocated");
    let (r, a) = count(|| expected_shortfall_ws(&pnl, 0.025, &mut work));
    assert!(r.is_ok());
    assert_eq!(a, 0, "expected_shortfall_ws allocated");
    let (r, a) = count(|| empirical_var_es_ws(&pnl, 0.01, &mut work));
    assert!(r.is_ok());
    assert_eq!(a, 0, "empirical_var_es_ws allocated");
    let (r, a) = count(|| var_bootstrap_se_ws(&pnl, 0.01, 64, 3, &mut work, &mut est));
    assert!(r.is_ok());
    assert_eq!(a, 0, "var_bootstrap_se_ws allocated");
    for tail in [TailModel::Normal, t] {
        let (r, a) = count(|| parametric_var_fast(&w, &cov, 0.01, tail));
        assert!(r.is_ok());
        assert_eq!(a, 0, "parametric_var_fast allocated");
        let (r, a) = count(|| parametric_es_fast(&w, &cov, 0.025, tail));
        assert!(r.is_ok());
        assert_eq!(a, 0, "parametric_es_fast allocated");
        let mut ws = McWorkspace::new(n, 2_000, 64);
        ws.prepare(&cov, tail).unwrap();
        let (r, a) = count(|| ws.run(&w, 0.01, 2_000, 9, 64));
        assert!(r.is_ok());
        assert_eq!(a, 0, "McWorkspace::run allocated");
        let (r, a) = count(|| ws.var(&w, 0.01, 2_000, 9));
        assert!(r.is_ok());
        assert_eq!(a, 0, "McWorkspace::var allocated");
    }
    // Error paths do not allocate either.
    let (r, a) = count(|| historical_var_ws(&pnl, 0.7, &mut work));
    assert_eq!(r, Err(VarFault::BadAlpha));
    assert_eq!(a, 0);
    let (r, a) = count(|| historical_var_ws(&pnl, 0.01, &mut work[..10]));
    assert_eq!(r, Err(VarFault::WorkspaceTooSmall));
    assert_eq!(a, 0);
    // The main API does allocate on the same input, so the counter works.
    let (_, a) = count(|| eq_var_engine::historical::historical_var(&pnl, 0.01));
    assert!(a > 0, "counter did not see the main API allocate");
}
