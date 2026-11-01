//! Kernel success paths do not touch the heap.
//!
//! A counting global allocator wraps the system one. This file holds a
//! single test so no other test thread allocates while a kernel runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use fx_options_engine::fast::{
    binomial_price_ws, gk_price_batch, gk_price_k, greeks_batch, greeks_k, implied_vol_k,
    mc_price_ws, BinomialWorkspace, GkBatch, GreeksBatchOut, McWorkspace,
};
use fx_options_engine::{gk_price, Exercise, OptionType};

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

fn count<R>(f: impl FnOnce() -> R) -> (usize, R) {
    let before = ALLOCS.load(Ordering::SeqCst);
    let r = black_box(f());
    (ALLOCS.load(Ordering::SeqCst) - before, r)
}

#[test]
fn kernels_do_not_allocate_on_success() {
    let (s, k, t, rd, rf, v) = (1.10, 1.12, 0.5, 0.0425, 0.0290, 0.0925);
    let call = OptionType::Call;

    let (n, r) = count(|| gk_price_k(s, k, t, rd, rf, v, call));
    assert!(r.is_ok());
    assert_eq!(n, 0, "gk_price_k allocated");

    let (n, r) = count(|| greeks_k(s, k, t, rd, rf, v, call));
    assert!(r.is_ok());
    assert_eq!(n, 0, "greeks_k allocated");

    let px = gk_price(s, k, t, rd, rf, v, call).unwrap();
    let (n, r) = count(|| implied_vol_k(px, s, k, t, rd, rf, call));
    assert!(r.is_ok());
    assert_eq!(n, 0, "implied_vol_k allocated");

    // Error paths of the kernel API do not allocate either.
    let (n, r) = count(|| gk_price_k(f64::NAN, k, t, rd, rf, v, call));
    assert!(r.is_err());
    assert_eq!(n, 0, "gk_price_k error path allocated");

    let sv = [s; 64];
    let kv: Vec<f64> = (0..64).map(|i| 0.9 + 0.01 * i as f64).collect();
    let tv = [t; 64];
    let rdv = [rd; 64];
    let rfv = [rf; 64];
    let vv = [v; 64];
    let ty = [call; 64];
    let b = GkBatch { s: &sv, k: &kv, t: &tv, r_d: &rdv, r_f: &rfv, sigma: &vv, option_type: &ty };
    let mut out = [0.0; 64];
    let (n, r) = count(|| gk_price_batch(&b, &mut out));
    assert!(r.is_ok());
    assert_eq!(n, 0, "gk_price_batch allocated");

    let mut g = [[0.0; 64]; 10];
    let [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] = &mut g;
    let mut go = GreeksBatchOut {
        price: a0,
        delta_spot: a1,
        delta_forward: a2,
        gamma: a3,
        vega: a4,
        theta: a5,
        rho_domestic: a6,
        rho_foreign: a7,
        vanna: a8,
        volga: a9,
    };
    let (n, r) = count(|| greeks_batch(&b, &mut go));
    assert!(r.is_ok());
    assert_eq!(n, 0, "greeks_batch allocated");

    let mut bws = BinomialWorkspace::with_steps(500);
    let (n, r) = count(|| {
        binomial_price_ws(&mut bws, s, k, t, rd, rf, v, OptionType::Put, 500, Exercise::American)
    });
    assert!(r.is_ok());
    assert_eq!(n, 0, "binomial_price_ws allocated");

    let mut mws = McWorkspace::with_paths(10_000);
    let (n, r) = count(|| mc_price_ws(&mut mws, s, k, t, rd, rf, v, call, 10_000, 7, true, true));
    assert!(r.is_ok());
    assert_eq!(n, 0, "mc_price_ws allocated");

    // Sanity: the counter does see the classic tree, which builds Vecs.
    let (n, _) = count(|| {
        fx_options_engine::binomial_price(s, k, t, rd, rf, v, OptionType::Put, 500, Exercise::American)
    });
    assert!(n > 0, "counter did not observe a known allocation");
}
