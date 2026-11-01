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
/// After a job, idle workers spin for a short while (`spin`) before they
/// block on a condition variable, so back-to-back calls do not pay a
/// kernel wake-up. No allocation happens in `run`, and the task function is
/// a plain function pointer plus a context pointer (no std::function).
/// One job runs at a time per pool; concurrent callers are serialised.

#ifndef EQOPT_THREAD_POOL_HPP
#define EQOPT_THREAD_POOL_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace eqopt {

class ThreadPool {
public:
    using TaskFn = void (*)(void* ctx, unsigned task) noexcept;

    /// \brief Start `workers` background threads (0 is allowed: every job
    ///        then runs on the calling thread).
    /// \param spin How long an idle worker polls for the next job before it
    ///        blocks.
    explicit ThreadPool(unsigned workers,
                        std::chrono::microseconds spin = std::chrono::microseconds(50));
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Number of background threads (the caller is one more).
    [[nodiscard]] unsigned workers() const noexcept {
        return static_cast<unsigned>(threads_.size());
    }

    /// \brief Run `fn(ctx, t)` for t in [0, n_tasks); blocks until done.
    void run(unsigned n_tasks, TaskFn fn, void* ctx) noexcept;

    /// \brief Convenience form for a callable `f(unsigned)` that does not
    ///        throw. `f` must outlive the call (it does: run blocks).
    template <class F>
    void for_each_task(unsigned n_tasks, F& f) noexcept {
        run(n_tasks, [](void* c, unsigned t) noexcept { (*static_cast<F*>(c))(t); }, &f);
    }

private:
    void worker_loop() noexcept;
    void drain(TaskFn fn, void* ctx, unsigned n) noexcept;

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

}  // namespace eqopt

#endif  // EQOPT_THREAD_POOL_HPP
