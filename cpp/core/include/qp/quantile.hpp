// qp/quantile.hpp: O(n) order statistics and the type-7 quantile.
//
// std::nth_element is O(n) on average (introselect: Musser 1997, worst case
// O(n log n)). It places the k-th smallest element exactly where a full sort
// would, so the value returned here is the same double a sort-based
// quantile returns: no tolerance applies. The interpolated type-7 quantile
// (Hyndman and Fan 1996, definition 7; NumPy's default) needs x(lo) and
// x(lo + 1); after nth_element x(lo + 1) is the minimum of the upper part,
// one more linear pass. The arithmetic x(lo) + f (x(lo + 1) - x(lo)) is the
// one the sort-based code uses, so the bits agree.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>

#include "qp/config.hpp"

namespace qp {

/// The k-th smallest value (0-based) of `work`, which is reordered.
/// Precondition: k < work.size() and no NaN in `work`.
[[nodiscard]] inline double select_kth(std::span<double> work, std::size_t k) noexcept
    QP_PRE(k < work.size()) {
    const auto kth = work.begin() + static_cast<std::ptrdiff_t>(k);
    std::nth_element(work.begin(), kth, work.end());
    return *kth;
}

/// Type-7 quantile of `work` (reordered) at probability q in [0, 1]:
/// h = (n - 1) q, lo = floor(h), x(lo) + (h - lo) (x(lo + 1) - x(lo)).
/// Precondition: non-empty, no NaN, 0 <= q <= 1.
[[nodiscard]] inline double quantile_type7(std::span<double> work, double q) noexcept
    QP_PRE(!work.empty()) {
    const std::size_t n = work.size();
    const double h = static_cast<double>(n - 1) * q;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    const double xlo = select_kth(work, lo);
    if (lo + 1 >= n) return xlo;
    const double xhi =
        *std::min_element(work.begin() + static_cast<std::ptrdiff_t>(lo) + 1, work.end());
    return xlo + (h - static_cast<double>(lo)) * (xhi - xlo);
}

}  // namespace qp
