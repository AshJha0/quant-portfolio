// C20. Constraining a template with a concept; C23. a padded counter.
#pragma once
#include <concepts>
#include <cstdint>
#include <ranges>

namespace cp {

// Anything we can iterate whose elements add together and value-initialise.
template <class R>
concept SummableRange = std::ranges::input_range<R> && requires(std::ranges::range_value_t<R> a) {
    { a + a } -> std::convertible_to<std::ranges::range_value_t<R>>;
    std::ranges::range_value_t<R>{};
};

template <SummableRange R>
auto sum_all(const R& r) {
    std::ranges::range_value_t<R> acc{};
    for (const auto& x : r) acc = acc + x;
    return acc;
}

// One counter per cache line, so threads bumping neighbouring counters do
// not invalidate each other's line (false sharing). 64 bytes is the line
// size on current x86-64 and most Arm server cores.
struct alignas(64) PaddedCounter {
    std::uint64_t value = 0;
};

}  // namespace cp
