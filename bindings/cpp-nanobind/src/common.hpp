// Shared helpers for the qp_cpp bindings.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace qpb {

namespace nb = nanobind;

// 1-D float64 input. C-contiguous float64 numpy arrays are used in place
// (no copy); anything else is converted by nanobind first.
using InVec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using InBool = nb::ndarray<const bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using InMat = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutVec = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

inline std::span<const double> span_of(const InVec& a) { return {a.data(), a.shape(0)}; }

// Fresh numpy array of n doubles owned by a capsule.
struct Owned {
    double* data;
    OutVec array;
};
inline Owned new_array(std::size_t n) {
    double* p = new double[n > 0 ? n : 1];
    nb::capsule owner(p, [](void* q) noexcept { delete[] static_cast<double*>(q); });
    return {p, OutVec(p, {n}, owner)};
}

[[noreturn]] inline void value_error(const std::string& msg) { throw nb::value_error(msg.c_str()); }

// "call" / "put" to a bool (true for call); anything else is a ValueError.
inline bool parse_call(const std::string& s) {
    if (s == "call") return true;
    if (s == "put") return false;
    value_error("option_type must be 'call' or 'put', got '" + s + "'");
}

// Run a throwing engine call and map std exceptions to ValueError with the
// engine's message (runtime_error is used by some engines for solver
// non-convergence; it is still a property of the inputs).
template <class F>
auto guarded(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const nb::python_error&) {
        throw;
    } catch (const nb::builtin_exception&) {
        throw;
    } catch (const std::exception& e) {
        value_error(e.what());
    }
}

inline void check_same_len(std::size_t n, std::size_t m, const char* name) {
    if (n != m)
        value_error(std::string("length mismatch: ") + name + " has " + std::to_string(m) +
                    " elements, expected " + std::to_string(n));
}

}  // namespace qpb
