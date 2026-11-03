// Shared helpers for the qp_cpp bindings.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
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

// ------------------------------------------------------------------ errors
//
// Engine failures raise the qp_core.errors classes (docs/VERSIONING.md):
//   InputError        (a ValueError)    bad arguments
//   ConvergenceError  (a RuntimeError)  a numerical method failed on
//                                       well-formed input; the engines'
//                                       throwing API signals these with
//                                       std::runtime_error
// When qp_core is not importable both fall back to ValueError, which is
// what the binding raised before. The lookup runs on the error path only
// and needs the GIL, which every caller below holds.

enum class ErrKind { Input, Convergence };

[[noreturn]] inline void raise_qp(ErrKind kind, const std::string& msg) {
    const char* name = kind == ErrKind::Input ? "InputError" : "ConvergenceError";
    nb::object cls;
    try {
        cls = nb::module_::import_("qp_core.errors").attr(name);
    } catch (const nb::python_error&) {
        throw nb::value_error(msg.c_str());  // qp_core missing: old behaviour
    }
    PyErr_SetString(cls.ptr(), msg.c_str());
    throw nb::python_error();
}

// Bad caller input. The name is kept from the first binding release.
[[noreturn]] inline void value_error(const std::string& msg) { raise_qp(ErrKind::Input, msg); }

// std::runtime_error (and its subclasses) is how the engines report solver
// and factorisation failures; every other std::exception is bad input.
[[noreturn]] inline void raise_std(const std::exception& e, const std::string& prefix = {}) {
    const bool numerical = dynamic_cast<const std::runtime_error*>(&e) != nullptr;
    raise_qp(numerical ? ErrKind::Convergence : ErrKind::Input, prefix + e.what());
}

// Run an engine's throw_error(e) and raise the mapped Python error. Using
// the engine's own thrower keeps one classification table, in the engine.
template <class Thrower>
[[noreturn]] void raise_engine(Thrower&& thrower, const std::string& prefix = {}) {
    try {
        thrower();
    } catch (const std::exception& e) {
        raise_std(e, prefix);
    }
    std::terminate();  // throw_error is [[noreturn]]; never reached
}

// "call" / "put" to a bool (true for call); anything else is a ValueError.
inline bool parse_call(const std::string& s) {
    if (s == "call") return true;
    if (s == "put") return false;
    value_error("option_type must be 'call' or 'put', got '" + s + "'");
}

// Run a throwing engine call and map std exceptions to the qp_core.errors
// classes with the engine's message (raise_std).
template <class F>
auto guarded(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const nb::python_error&) {
        throw;
    } catch (const nb::builtin_exception&) {
        throw;
    } catch (const std::exception& e) {
        raise_std(e);
    }
}

inline void check_same_len(std::size_t n, std::size_t m, const char* name) {
    if (n != m)
        value_error(std::string("length mismatch: ") + name + " has " + std::to_string(m) +
                    " elements, expected " + std::to_string(n));
}

}  // namespace qpb
