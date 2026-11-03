// C12. A minimal move-only owning pointer (the core of std::unique_ptr).
#pragma once
#include <utility>

namespace cp {

template <class T>
class UniquePtr {
public:
    UniquePtr() noexcept = default;
    explicit UniquePtr(T* p) noexcept : p_(p) {}
    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;
    UniquePtr(UniquePtr&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
    UniquePtr& operator=(UniquePtr&& o) noexcept {
        // self-move safe: reset(release()) on the same object keeps the pointer
        reset(o.release());
        return *this;
    }
    ~UniquePtr() { delete p_; }

    T* release() noexcept { return std::exchange(p_, nullptr); }
    void reset(T* p = nullptr) noexcept {
        T* old = std::exchange(p_, p);  // assign first, then delete: safe if ~T re-enters
        delete old;
    }
    T* get() const noexcept { return p_; }
    T& operator*() const noexcept { return *p_; }
    T* operator->() const noexcept { return p_; }
    explicit operator bool() const noexcept { return p_ != nullptr; }

private:
    T* p_ = nullptr;
};

template <class T, class... A>
UniquePtr<T> make_unique(A&&... a) {
    return UniquePtr<T>(new T(std::forward<A>(a)...));
}

}  // namespace cp
