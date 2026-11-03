// C17. Fixed-size thread pool returning futures.
#pragma once
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "blocking_queue.hpp"

namespace cp {

class ThreadPool {
public:
    explicit ThreadPool(unsigned n) {
        for (unsigned i = 0; i < n; ++i)
            workers_.emplace_back([this] {
                while (auto job = jobs_.pop()) (*job)();
            });
    }
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    // Closing lets the workers drain the queue and exit; the jthreads then
    // join in their destructors, which run before jobs_ is destroyed.
    ~ThreadPool() { jobs_.close(); }

    template <class F>
    auto submit(F f) -> std::future<std::invoke_result_t<F>> {
        // packaged_task is move-only and std::function needs a copyable target, so share it
        auto task = std::make_shared<std::packaged_task<std::invoke_result_t<F>()>>(std::move(f));
        auto fut = task->get_future();
        jobs_.push([task] { (*task)(); });  // an exception is stored in the future
        return fut;
    }

private:
    BlockingQueue<std::function<void()>> jobs_;
    std::vector<std::jthread> workers_;  // declared last, so destroyed (joined) first
};

}  // namespace cp
