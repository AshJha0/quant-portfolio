//! loom model checks. Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p qp-ull-ring --release --test loom`
//! Without `--cfg loom` this file compiles to nothing.
#![cfg(loom)]

use qp_ull_ring::channel;

/// Every interleaving of 3 pushes against 3 pops through a capacity-2 ring
/// delivers 0, 1, 2 in order, exactly once, with the full-ring path taken
/// (the producer must observe a consumer `head` store to make progress).
#[test]
fn loom_fifo_through_full_ring() {
    let mut b = loom::model::Builder::new();
    b.preemption_bound = Some(3);
    b.check(|| {
        let (mut p, mut c) = channel::<u32>(2);
        let t = loom::thread::spawn(move || {
            for i in 0..3 {
                let mut v = i;
                while let Err(back) = p.push(v) {
                    v = back;
                    loom::thread::yield_now();
                }
            }
        });
        let mut got = Vec::new();
        while got.len() < 3 {
            match c.pop() {
                Some(v) => got.push(v),
                None => loom::thread::yield_now(),
            }
        }
        t.join().unwrap();
        assert_eq!(got, vec![0, 1, 2]);
        assert_eq!(c.pop(), None);
    });
}

/// A payload written before publish is visible after pop.
///
/// Mutation check, run by hand on 2026-11-03 with loom 0.7.2: weakening the
/// producer's `tail` store, or the consumer's `head` store, from Release to
/// Relaxed makes `loom_fifo_through_full_ring` fail with "Causality
/// violation: Concurrent write accesses to `UnsafeCell`". With the `tail`
/// mutation this test does not terminate (loom keeps serving the stale
/// value to the spin loop until the stack overflows), which is also a fail.
#[test]
fn loom_payload_visibility() {
    loom::model(|| {
        let (mut p, mut c) = channel::<[u64; 2]>(1);
        let t = loom::thread::spawn(move || {
            p.push([7, 9]).unwrap();
        });
        loop {
            if let Some(v) = c.pop() {
                assert_eq!(v, [7, 9]);
                break;
            }
            loom::thread::yield_now();
        }
        t.join().unwrap();
    });
}

/// Items left in the ring when both handles drop are dropped exactly once.
#[test]
fn loom_drop_with_items_in_flight() {
    loom::model(|| {
        let marker = loom::sync::Arc::new(());
        let (mut p, mut c) = channel(2);
        let m2 = marker.clone();
        let t = loom::thread::spawn(move || {
            let _ = p.push(m2);
        });
        let _ = c.pop();
        t.join().unwrap();
        drop(c);
        assert_eq!(loom::sync::Arc::strong_count(&marker), 1);
    });
}
