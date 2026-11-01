//! Fixed-partition data parallelism on `std::thread::scope`.
//!
//! The output slice is cut into contiguous chunks of whole rows. Each chunk
//! is written by exactly one thread and every element depends only on its
//! own row, so the result is the same for any thread count.

/// Worker threads to use by default: the machine's available parallelism.
pub(crate) fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Run `f(first_row, chunk)` over `out`, split into at most `threads`
/// contiguous chunks of whole rows (`row_len` elements each). A chunk never
/// holds fewer than `min_rows` rows, so small inputs stay on the caller's
/// thread. Threads are created once per call inside one scope.
pub(crate) fn for_row_chunks<T, F>(out: &mut [T], row_len: usize, min_rows: usize, threads: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    let row_len = row_len.max(1);
    let n_rows = out.len() / row_len;
    let threads = threads.max(1);
    let rows_per = n_rows.div_ceil(threads).max(min_rows.max(1));
    if threads == 1 || rows_per >= n_rows {
        f(0, out);
        return;
    }
    let f = &f;
    std::thread::scope(|s| {
        let mut chunks = out.chunks_mut(rows_per * row_len).enumerate();
        // The caller's thread takes the first chunk; the rest get one
        // scoped thread each.
        let first = chunks.next();
        for (c, chunk) in chunks {
            s.spawn(move || f(c * rows_per, chunk));
        }
        if let Some((_, chunk)) = first {
            f(0, chunk);
        }
    });
}
