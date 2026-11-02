//! Fixed-partition data parallelism on `std::thread::scope`.
//!
//! The helper lives in `qp_core::par`: the output slice is cut into
//! contiguous chunks of whole rows, each written by one thread, so the
//! result is the same for any thread count.

pub(crate) use qp_core::par::{default_threads, for_row_chunks};
