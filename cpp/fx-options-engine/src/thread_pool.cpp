#include "fxopt/thread_pool.hpp"

#include <algorithm>

namespace fxopt {

ThreadPool::ThreadPool(unsigned workers, std::chrono::microseconds spin) : spin_(spin) {
    threads_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) threads_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        stop_ = true;
        generation_.fetch_add(1, std::memory_order_release);
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
}

void ThreadPool::drain(TaskFn fn, void* ctx, unsigned n) noexcept {
    for (unsigned t = next_.fetch_add(1, std::memory_order_relaxed); t < n;
         t = next_.fetch_add(1, std::memory_order_relaxed)) {
        fn(ctx, t);
        if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            std::lock_guard<std::mutex> lk(mutex_);
            done_cv_.notify_all();
        }
    }
}

void ThreadPool::worker_loop() noexcept {
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
            wake_.wait(lk, [&] {
                return stop_ || generation_.load(std::memory_order_acquire) != seen;
            });
            if (stop_) return;
            seen = generation_.load(std::memory_order_acquire);
            fn = fn_;
            ctx = ctx_;
            n = n_tasks_;
            ++active_;
        }
        drain(fn, ctx, n);
        std::lock_guard<std::mutex> lk(mutex_);
        if (--active_ == 0) done_cv_.notify_all();
    }
}

void ThreadPool::run(unsigned n_tasks, TaskFn fn, void* ctx) noexcept {
    if (n_tasks == 0) return;
    if (threads_.empty() || n_tasks == 1) {
        for (unsigned t = 0; t < n_tasks; ++t) fn(ctx, t);
        return;
    }
    std::lock_guard<std::mutex> job(run_mutex_);
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
    done_cv_.wait(lk, [&] {
        return done_.load(std::memory_order_acquire) == n_tasks && active_ == 0;
    });
}

ThreadPool& default_thread_pool() {
    static ThreadPool pool(std::max(1U, std::thread::hardware_concurrency()) > 1
                               ? std::thread::hardware_concurrency() - 1
                               : 1U);
    return pool;
}

}  // namespace fxopt
