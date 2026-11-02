/// \file thread_pool.hpp
/// \brief Fixed-size persistent worker pool for the Monte Carlo kernels.
///
/// Threads are created once, in the constructor, and live until the pool is
/// destroyed. `run(n, fn, ctx)` executes `fn(ctx, t)` for every task index
/// `t` in `[0, n)` and returns when all tasks are done; the calling thread
/// takes tasks too. Tasks are claimed from an atomic counter, so which
/// thread runs a task varies, but each task index always does the same work
/// on the same data. Kernels that give every task its own RNG stream and
/// output range therefore return identical bits whatever the pool size.
///
/// After a job, idle workers may spin for a short while (`spin`) before they
/// block on a condition variable, so back-to-back calls do not pay a
/// kernel wake-up. The constructor default is no spin, which suits a shared
/// host; `default_thread_pool()` spins for 50 us because the Monte Carlo
/// kernels call it back to back. No allocation happens in `run`, and the
/// task function is a plain function pointer plus a context pointer (no
/// std::function). One job runs at a time per pool; concurrent callers are
/// serialised.
///
/// Contract:
/// - Tasks must not throw. `TaskFn` is `noexcept`, so an exception that
///   escapes a task calls `std::terminate`. Catch inside the task and
///   report through the context if a task can fail.
/// - A task may call `run` on the same pool (for example through
///   `default_thread_pool()`). The nested job then runs inline on the
///   calling thread instead of deadlocking on the job lock.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "qp/config.hpp"

namespace qp {

class ThreadPool {
public:
    using TaskFn = void (*)(void* ctx, unsigned task) noexcept;

    /// \brief Start `workers` background threads (0 is allowed: every job
    ///        then runs on the calling thread).
    /// \param spin How long an idle worker polls for the next job before it
    ///        blocks.
    explicit ThreadPool(unsigned workers,
                        std::chrono::microseconds spin = std::chrono::microseconds(0));
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Number of background threads (the caller is one more).
    [[nodiscard]] unsigned workers() const noexcept {
        return static_cast<unsigned>(threads_.size());
    }

    /// \brief Run `fn(ctx, t)` for t in [0, n_tasks); blocks until done.
    void run(unsigned n_tasks, TaskFn fn, void* ctx) noexcept QP_PRE(n_tasks == 0 || fn != nullptr);

    /// \brief Convenience form for a callable `f(unsigned)` that does not
    ///        throw. `f` must outlive the call (it does: run blocks).
    template <class F>
    void for_each_task(unsigned n_tasks, F& f) noexcept {
        run(n_tasks, [](void* c, unsigned t) noexcept { (*static_cast<F*>(c))(t); }, &f);
    }

private:
    void worker_loop() noexcept;
    void drain(TaskFn fn, void* ctx, unsigned n) noexcept;
    static void run_inline(unsigned n_tasks, TaskFn fn, void* ctx) noexcept;

    // The pool whose task the current thread is executing, if any. Used to
    // detect a nested run() on the same pool.
    static const ThreadPool*& current_pool() noexcept {
        thread_local const ThreadPool* p = nullptr;
        return p;
    }
    struct InPool {
        const ThreadPool* saved;
        explicit InPool(const ThreadPool* p) noexcept : saved(current_pool()) {
            current_pool() = p;
        }
        ~InPool() { current_pool() = saved; }
        InPool(const InPool&) = delete;
        InPool& operator=(const InPool&) = delete;
    };

    std::vector<std::thread> threads_;
    std::chrono::microseconds spin_;
    std::mutex run_mutex_;  // one job at a time
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_cv_;
    std::atomic<std::uint64_t> generation_{0};
    std::atomic<unsigned> next_{0};
    std::atomic<unsigned> done_{0};
    unsigned n_tasks_ = 0;
    TaskFn fn_ = nullptr;
    void* ctx_ = nullptr;
    unsigned active_ = 0;  // workers inside drain(), guarded by mutex_
    bool stop_ = false;
};

/// \brief Process-wide pool used by the throwing mc_price() when
///        `threads > 1`. Created on first use with
///        `std::thread::hardware_concurrency() - 1` workers (at least 1).
ThreadPool& default_thread_pool();

// ------------------------------------------------------------ definitions
inline ThreadPool::ThreadPool(unsigned workers, std::chrono::microseconds spin) : spin_(spin) {
    threads_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) threads_.emplace_back([this] { worker_loop(); });
}

inline ThreadPool::~ThreadPool() {
    {
        std::scoped_lock lk(mutex_);
        stop_ = true;
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
}

inline void ThreadPool::run_inline(unsigned n_tasks, TaskFn fn, void* ctx) noexcept {
    for (unsigned t = 0; t < n_tasks; ++t) fn(ctx, t);
}

inline void ThreadPool::drain(TaskFn fn, void* ctx, unsigned n) noexcept {
    const InPool in(this);
    for (unsigned t = next_.fetch_add(1, std::memory_order_relaxed); t < n;
         t = next_.fetch_add(1, std::memory_order_relaxed)) {
        fn(ctx, t);
        if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            std::scoped_lock lk(mutex_);
            done_cv_.notify_all();
        }
    }
}

inline void ThreadPool::worker_loop() noexcept {
    std::uint64_t seen = 0;
    TaskFn fn = nullptr;
    void* ctx = nullptr;
    unsigned n = 0;
    for (;;) {
        // Spin briefly for the next job, then block.
        const auto deadline = std::chrono::steady_clock::now() + spin_;
        while (generation_.load(std::memory_order_acquire) == seen &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        {
            std::unique_lock<std::mutex> lk(mutex_);
            wake_.wait(
                lk, [&] { return stop_ || generation_.load(std::memory_order_acquire) != seen; });
            if (stop_) return;
            seen = generation_.load(std::memory_order_acquire);
            fn = fn_;
            ctx = ctx_;
            n = n_tasks_;
            ++active_;
        }
        drain(fn, ctx, n);
        std::scoped_lock lk(mutex_);
        if (--active_ == 0) done_cv_.notify_all();
    }
}

inline void ThreadPool::run(unsigned n_tasks, TaskFn fn, void* ctx) noexcept {
    if (n_tasks == 0) return;
    if (threads_.empty() || n_tasks == 1 || current_pool() == this) {
        // No workers, one task, or a nested call from inside one of this
        // pool's tasks: run on the calling thread.
        run_inline(n_tasks, fn, ctx);
        return;
    }
    std::scoped_lock job(run_mutex_);
    {
        // A worker that woke late for the previous job may still be inside
        // drain() with that job's parameters; let it leave first.
        std::unique_lock<std::mutex> lk(mutex_);
        done_cv_.wait(lk, [&] { return active_ == 0; });
        fn_ = fn;
        ctx_ = ctx;
        n_tasks_ = n_tasks;
        done_.store(0, std::memory_order_relaxed);
        next_.store(0, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    drain(fn, ctx, n_tasks);
    // Wait until every task is done and no worker is still inside drain(),
    // so the next job cannot be observed half-set-up.
    std::unique_lock<std::mutex> lk(mutex_);
    done_cv_.wait(lk,
                  [&] { return done_.load(std::memory_order_acquire) == n_tasks && active_ == 0; });
}

inline ThreadPool& default_thread_pool() {
    static ThreadPool pool(std::max(1U, std::thread::hardware_concurrency()) > 1
                               ? std::thread::hardware_concurrency() - 1
                               : 1U,
                           std::chrono::microseconds(50));
    return pool;
}

}  // namespace qp
