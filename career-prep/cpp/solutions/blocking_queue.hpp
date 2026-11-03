// C14. Multi-producer multi-consumer blocking queue with close().
#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

namespace cp {

template <class T>
class BlockingQueue {
public:
    // Returns false if the queue was closed.
    bool push(T v) {
        {
            std::lock_guard lock(m_);
            if (closed_) return false;
            q_.push_back(std::move(v));
        }
        cv_.notify_one();  // notify after unlocking: the woken thread can take the lock at once
        return true;
    }

    // Blocks until an item arrives, or the queue is closed and drained.
    std::optional<T> pop() {
        std::unique_lock lock(m_);
        cv_.wait(lock, [this] { return closed_ || !q_.empty(); });  // the predicate absorbs spurious wake-ups
        if (q_.empty()) return std::nullopt;
        T v = std::move(q_.front());
        q_.pop_front();
        return v;
    }

    void close() {
        {
            std::lock_guard lock(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<T> q_;
    bool closed_ = false;
};

}  // namespace cp
