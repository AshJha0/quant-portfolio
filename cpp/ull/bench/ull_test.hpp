// Minimal self-contained test macros for the cpp/ull tests.
//
// The ull tests deliberately avoid a test framework dependency so the
// directory builds with nothing but a C++20 compiler and CMake. Each test
// binary returns non-zero when any check fails; CTest reads the exit code.
#pragma once

#include <cstdio>
#include <cstdlib>

namespace ull::test {
inline int& failures() {
    static int n = 0;
    return n;
}
}  // namespace ull::test

#define ULL_CHECK(cond)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__,        \
                         __LINE__, #cond);                                     \
            ++::ull::test::failures();                                         \
        }                                                                      \
    } while (0)

#define ULL_REQUIRE(cond)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "%s:%d: REQUIRE failed: %s\n", __FILE__,      \
                         __LINE__, #cond);                                     \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define ULL_TEST_MAIN_END()                                                    \
    do {                                                                       \
        if (::ull::test::failures() != 0) {                                    \
            std::fprintf(stderr, "%d check(s) failed\n",                       \
                         ::ull::test::failures());                             \
            return 1;                                                          \
        }                                                                      \
        std::printf("all checks passed\n");                                    \
        return 0;                                                              \
    } while (0)
