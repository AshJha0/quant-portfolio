//! A persistent worker pool, so a hot path does not spawn threads per call.
//!
//! [`crate::par`] runs on `std::thread::scope`, which creates and joins its
//! threads on every call (tens of microseconds). This pool starts its
//! workers once and feeds them boxed jobs over a channel. Jobs must be
//! `'static` because the crate forbids `unsafe` and a pool cannot borrow
//! the caller's stack safely, so inputs travel in an [`Arc`]. Results come
//! back tagged with their task index and are returned in index order: the
//! output does not depend on the number of workers or on scheduling.

use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A fixed set of worker threads.
#[derive(Debug)]
pub struct ThreadPool {
    tx: Option<Sender<Job>>,
    workers: Vec<JoinHandle<()>>,
}

impl ThreadPool {
    /// Start `workers` threads (at least one).
    #[must_use]
    pub fn new(workers: usize) -> Self {
        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let workers = (0..workers.max(1))
            .map(|_| {
                let rx = Arc::clone(&rx);
                std::thread::spawn(move || {
                    loop {
                        // A poisoned lock means another worker panicked while
                        // holding it; the receiver is still usable.
                        let job = match rx.lock() {
                            Ok(g) => g.recv(),
                            Err(p) => p.into_inner().recv(),
                        };
                        match job {
                            Ok(job) => job(),
                            Err(_) => break, // sender dropped: shut down
                        }
                    }
                })
            })
            .collect();
        Self {
            tx: Some(tx),
            workers,
        }
    }

    /// Number of worker threads.
    #[must_use]
    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// `[f(0), f(1), ..., f(n_tasks - 1)]`, computed on the workers.
    ///
    /// # Panics
    /// If a task panics (the panic is reported as a missing result).
    pub fn map<T, F>(&self, n_tasks: usize, f: Arc<F>) -> Vec<T>
    where
        T: Send + 'static,
        F: Fn(usize) -> T + Send + Sync + 'static,
    {
        let (rtx, rrx) = channel::<(usize, T)>();
        let tx = self.tx.as_ref().expect("pool is running");
        for i in 0..n_tasks {
            let f = Arc::clone(&f);
            let rtx = rtx.clone();
            tx.send(Box::new(move || {
                let _ = rtx.send((i, f(i)));
            }))
            .expect("workers are alive");
        }
        drop(rtx);
        let mut out: Vec<Option<T>> = (0..n_tasks).map(|_| None).collect();
        for (i, v) in rrx {
            out[i] = Some(v);
        }
        out.into_iter()
            .map(|v| v.expect("a pool task panicked"))
            .collect()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        drop(self.tx.take());
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::philox::PhiloxStream;

    #[test]
    fn results_independent_of_worker_count() {
        let work = Arc::new(|c: usize| {
            let mut s = PhiloxStream::new(2026, c as u64);
            (0..1000).map(|_| s.uniform()).sum::<f64>()
        });
        let serial: Vec<f64> = (0..64).map(|c| work(c)).collect();
        for w in [1, 3, 8] {
            let pool = ThreadPool::new(w);
            assert_eq!(pool.map(64, Arc::clone(&work)), serial);
            assert_eq!(pool.map(64, Arc::clone(&work)), serial); // reuse
        }
    }
}
