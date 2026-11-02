// qp/config.hpp: language-level switches shared by the core and the engines.
//
// The floor is C++23. A C++26 build (CMake option QP_CXX26) turns on the
// features below only when the compiler reports them through a feature-test
// macro, so the same source builds on every supported compiler and the
// results do not depend on the language level.
//
// Contracts (P2900). QP_PRE(cond) and QP_POST(r, cond) expand to
// `pre(cond)` and `post(r: cond)` when __cpp_contracts is defined and to
// nothing otherwise. They state programmer preconditions only: pointer and
// size agreements, a probability the caller has already checked. Market
// data is never validated by a contract, because whether a violated
// contract is checked depends on the build mode; the kernels validate
// market data themselves and report it through std::expected.

#pragma once

#if defined(__cpp_contracts) && !defined(QP_NO_CONTRACTS)
#define QP_HAS_CONTRACTS 1
#define QP_PRE(...) pre(__VA_ARGS__)
#define QP_POST(r, ...) post(r : __VA_ARGS__)
#else
#define QP_HAS_CONTRACTS 0
#define QP_PRE(...)
#define QP_POST(r, ...)
#endif
