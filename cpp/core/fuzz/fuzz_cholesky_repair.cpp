// Coverage-guided fuzz target for the validated Cholesky with jitter
// ladder and PSD repair (qp/matrix.hpp), used by both VaR engines.
//
// Input: n in 1..6, n volatilities in [1e-4, 10], n(n-1)/2 raw
// correlations in [-1.5, 1.5] (so many matrices are indefinite), and a
// repair kind. Properties:
//   1. qp::cholesky either returns or throws std::invalid_argument /
//      std::runtime_error; nothing else, no sanitizer report.
//   2. On return, L is lower triangular and finite, and L L^T reproduces
//      the factored matrix (the input, or its repair, plus jitter * I) to
//      1e-8 of the largest variance.
//   3. A repaired matrix is symmetric and keeps the input's variances
//      (both repair kinds restore the diagonal) to 1e-8 relative.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "fuzz_input.hpp"
#include "qp/matrix.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    qp_fuzz::FuzzInput in(data, size);
    const std::size_t n = 1 + static_cast<std::size_t>(in.u64() % 6);
    const auto kind = static_cast<qp::linalg::RepairKind>(in.u64() % 3);
    double vol[6];
    for (std::size_t i = 0; i < n; ++i) vol[i] = in.in(1e-4, 10.0);
    qp::Matrix cov(n, n);
    double vmax = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cov(i, i) = vol[i] * vol[i];
        vmax = std::fmax(vmax, cov(i, i));
        for (std::size_t j = 0; j < i; ++j) {
            const double c = in.in(-1.5, 1.5) * vol[i] * vol[j];
            cov(i, j) = c;
            cov(j, i) = c;
        }
    }

    qp::linalg::CholeskyOut<qp::Matrix> out;
    try {
        out = qp::linalg::cholesky(cov, kind, "fuzz");
    } catch (const std::invalid_argument&) {
        return 0;
    } catch (const std::runtime_error&) {
        return 0;
    }

    qp::Matrix target = cov;
    if (out.repaired) {
        target = qp::linalg::repair_covariance(cov, kind);
        for (std::size_t i = 0; i < n; ++i) {
            if (!(std::fabs(target(i, i) - cov(i, i)) <= 1e-8 * cov(i, i))) std::abort();
            for (std::size_t j = 0; j < i; ++j) {
                if (target(i, j) != target(j, i)) std::abort();
            }
        }
    }
    const qp::Matrix& l = out.lower;
    if (l.rows() != n || l.cols() != n) std::abort();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(l(i, j))) std::abort();
            if (j > i && l(i, j) != 0.0) std::abort();
            double s = 0.0;
            for (std::size_t k = 0; k <= (i < j ? i : j); ++k) s += l(i, k) * l(j, k);
            const double want = target(i, j) + (i == j ? out.jitter : 0.0);
            if (!(std::fabs(s - want) <= 1e-8 * vmax)) std::abort();
        }
    }
    return 0;
}
