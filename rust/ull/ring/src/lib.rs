//! Bounded single-producer single-consumer (SPSC) ring.
//!
//! Plain idea: one thread writes, one thread reads, and each side owns one
//! counter. The writer publishes a slot by storing its `tail` with Release;
//! the reader sees the slot after loading `tail` with Acquire. Nothing else
//! is shared, so no lock and no compare-and-swap are needed (Lamport 1983,
//! "Specifying concurrent program modules", the original single-writer
//! single-reader queue).
//!
//! Formal statement: with capacity `N` (a power of two), `head` and `tail`
//! are free-running `usize` counters and `0 <= tail - head <= N` (wrapping)
//! holds at every point. Slot `i & (N-1)` belongs to the consumer between the
//! Release store of `tail > i` and the Release store of `head > i`, and to the
//! producer otherwise.
//!
//! Each side also keeps a private cached copy of the other side's counter
//! and only re-reads the shared atomic when the cache says the ring looks
//! full (producer) or empty (consumer). This is the cached-index refinement
//! described by Vyukov ("Single-producer/single-consumer queue",
//! 1024cores.net). In the steady state it cuts cross-core traffic to one
//! cache-line transfer per batch rather than per message.
//!
//! Worked example: capacity 4; push 1, 2, 3; pop gives 1; push 4, 5. Now
//! `tail` is 5 and `head` is 1, so `tail - head == 4` and the next push
//! returns `Err` with the value handed back.
//!
//! Hot-path properties: `push`/`pop` never allocate, never block and never
//! spin; the caller chooses the wait policy (busy-poll in the bench).
//!
//! Model checking: run
//! `RUSTFLAGS="--cfg loom" cargo test -p qp-ull-ring --release --test loom`.
//! loom explores the interleavings allowed by the C11 memory model (it
//! implements the CDSChecker approach, Norris and Demsky 2013); it is the
//! Rust counterpart of the GenMC check named for the C++ lane.

#![warn(missing_docs)]

mod sync {
    #[cfg(loom)]
    pub(crate) use loom::cell::UnsafeCell;
    #[cfg(loom)]
    pub(crate) use loom::sync::Arc;
    #[cfg(loom)]
    pub(crate) use loom::sync::atomic::{AtomicUsize, Ordering};

    #[cfg(not(loom))]
    pub(crate) use std::sync::Arc;
    #[cfg(not(loom))]
    pub(crate) use std::sync::atomic::{AtomicUsize, Ordering};

    /// Thin wrapper so the std build and the loom build share one call shape.
    #[cfg(not(loom))]
    #[derive(Debug)]
    pub(crate) struct UnsafeCell<T>(std::cell::UnsafeCell<T>);

    #[cfg(not(loom))]
    impl<T> UnsafeCell<T> {
        pub(crate) fn new(v: T) -> Self {
            Self(std::cell::UnsafeCell::new(v))
        }
        #[inline(always)]
        pub(crate) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
            f(self.0.get())
        }
    }
}

use std::mem::MaybeUninit;
use sync::{Arc, AtomicUsize, Ordering, UnsafeCell};

/// Pads and aligns a value to 128 bytes: two 64-byte lines, because the
/// adjacent-line (spatial) prefetcher on Intel cores fetches lines in pairs,
/// so 64-byte padding can still false-share there (Intel 64 and IA-32
/// Architectures Optimization Reference Manual, hardware prefetching).
#[repr(align(128))]
struct CachePadded<T>(T);

struct Shared<T> {
    /// Consumer-owned counter; read by the producer.
    head: CachePadded<AtomicUsize>,
    /// Producer-owned counter; read by the consumer.
    tail: CachePadded<AtomicUsize>,
    mask: usize,
    slots: Box<[UnsafeCell<MaybeUninit<T>>]>,
}

// SAFETY: the protocol hands each slot to exactly one side at a time, and
// values of `T` cross threads, so `T: Send` is required and sufficient.
unsafe impl<T: Send> Send for Shared<T> {}
// SAFETY: as above; shared access is only through the atomics.
unsafe impl<T: Send> Sync for Shared<T> {}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Acquire);
        let mut i = head;
        while i != tail {
            // SAFETY: slots in [head, tail) hold initialised values that no
            // one else can reach once both handles are gone.
            self.slots[i & self.mask].with_mut(|p| unsafe { (*p).assume_init_drop() });
            i = i.wrapping_add(1);
        }
    }
}

/// Producer handle. Exactly one exists per ring; it is `Send` but not `Clone`.
pub struct Producer<T> {
    shared: Arc<Shared<T>>,
    tail: usize,
    head_cache: usize,
}

/// Consumer handle. Exactly one exists per ring; it is `Send` but not `Clone`.
pub struct Consumer<T> {
    shared: Arc<Shared<T>>,
    head: usize,
    tail_cache: usize,
}

/// Creates a ring that holds `capacity` items. `capacity` is rounded up to a
/// power of two (minimum 1). All storage is allocated here, once.
pub fn channel<T: Send>(capacity: usize) -> (Producer<T>, Consumer<T>) {
    let cap = capacity.max(1).next_power_of_two();
    let slots: Box<[_]> = (0..cap)
        .map(|_| UnsafeCell::new(MaybeUninit::uninit()))
        .collect();
    let shared = Arc::new(Shared {
        head: CachePadded(AtomicUsize::new(0)),
        tail: CachePadded(AtomicUsize::new(0)),
        mask: cap - 1,
        slots,
    });
    (
        Producer {
            shared: shared.clone(),
            tail: 0,
            head_cache: 0,
        },
        Consumer {
            shared,
            head: 0,
            tail_cache: 0,
        },
    )
}

impl<T> Producer<T> {
    /// Ring capacity (a power of two).
    pub fn capacity(&self) -> usize {
        self.shared.mask + 1
    }

    /// Pushes `v`, or gives it back when the ring is full.
    #[inline]
    pub fn push(&mut self, v: T) -> Result<(), T> {
        let cap = self.shared.mask + 1;
        if self.tail.wrapping_sub(self.head_cache) == cap {
            // Acquire pairs with the consumer's Release store of `head`: the
            // consumer has finished reading the slot we are about to reuse.
            self.head_cache = self.shared.head.0.load(Ordering::Acquire);
            if self.tail.wrapping_sub(self.head_cache) == cap {
                return Err(v);
            }
        }
        let idx = self.tail & self.shared.mask;
        // SAFETY: slot `idx` is outside [head, tail), so only we touch it.
        self.shared.slots[idx].with_mut(|p| unsafe {
            (*p).write(v);
        });
        self.tail = self.tail.wrapping_add(1);
        // Release publishes the slot write above to the consumer.
        self.shared.tail.0.store(self.tail, Ordering::Release);
        Ok(())
    }
}

impl<T> Consumer<T> {
    /// Ring capacity (a power of two).
    pub fn capacity(&self) -> usize {
        self.shared.mask + 1
    }

    /// Pops the oldest item, or `None` when the ring is empty.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.head == self.tail_cache {
            // Acquire pairs with the producer's Release store of `tail`.
            self.tail_cache = self.shared.tail.0.load(Ordering::Acquire);
            if self.head == self.tail_cache {
                return None;
            }
        }
        let idx = self.head & self.shared.mask;
        // SAFETY: slot `idx` is in [head, tail) and was published by Release.
        let v = self.shared.slots[idx].with_mut(|p| unsafe { (*p).assume_init_read() });
        self.head = self.head.wrapping_add(1);
        // Release hands the slot back to the producer after our read.
        self.shared.head.0.store(self.head, Ordering::Release);
        Some(v)
    }

    /// Number of items visible to the consumer right now.
    pub fn len(&mut self) -> usize {
        self.tail_cache = self.shared.tail.0.load(Ordering::Acquire);
        self.tail_cache.wrapping_sub(self.head)
    }

    /// True when no item is visible right now.
    pub fn is_empty(&mut self) -> bool {
        self.len() == 0
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn worked_example_capacity_four() {
        let (mut p, mut c) = channel::<u32>(4);
        assert_eq!(p.capacity(), 4);
        for v in [1, 2, 3] {
            p.push(v).unwrap();
        }
        assert_eq!(c.pop(), Some(1));
        p.push(4).unwrap();
        p.push(5).unwrap();
        assert_eq!(p.push(6), Err(6));
        assert_eq!(c.len(), 4);
        let drained: Vec<_> = std::iter::from_fn(|| c.pop()).collect();
        assert_eq!(drained, vec![2, 3, 4, 5]);
        assert!(c.is_empty());
    }

    #[test]
    fn capacity_rounds_up() {
        assert_eq!(channel::<u8>(0).0.capacity(), 1);
        assert_eq!(channel::<u8>(5).1.capacity(), 8);
    }

    #[test]
    fn drops_unconsumed_items_exactly_once() {
        let marker = std::sync::Arc::new(());
        {
            let (mut p, mut c) = channel(8);
            for _ in 0..5 {
                p.push(marker.clone()).unwrap();
            }
            drop(c.pop());
            assert_eq!(std::sync::Arc::strong_count(&marker), 5);
        }
        assert_eq!(std::sync::Arc::strong_count(&marker), 1);
    }

    #[test]
    fn counters_wrap_without_loss() {
        let (mut p, mut c) = channel::<usize>(2);
        let start = usize::MAX - 3;
        p.tail = start;
        p.head_cache = start;
        c.head = start;
        c.tail_cache = start;
        p.shared.head.0.store(start, Ordering::Relaxed);
        p.shared.tail.0.store(start, Ordering::Relaxed);
        for i in 0..10 {
            p.push(i).unwrap();
            p.push(i + 100).unwrap();
            assert!(p.push(0).is_err());
            assert_eq!(c.pop(), Some(i));
            assert_eq!(c.pop(), Some(i + 100));
            assert_eq!(c.pop(), None);
        }
    }

    #[test]
    fn two_threads_preserve_order_and_count() {
        const N: u64 = 2_000_000;
        let (mut p, mut c) = channel::<u64>(1024);
        let t = std::thread::spawn(move || {
            for i in 0..N {
                let mut v = i;
                while let Err(back) = p.push(v) {
                    v = back;
                    std::hint::spin_loop();
                }
            }
        });
        let mut expect = 0;
        let mut sum = 0u64;
        while expect < N {
            if let Some(v) = c.pop() {
                assert_eq!(v, expect);
                sum += v;
                expect += 1;
            } else {
                std::hint::spin_loop();
            }
        }
        t.join().unwrap();
        assert_eq!(sum, N * (N - 1) / 2);
        assert_eq!(c.pop(), None);
    }
}
