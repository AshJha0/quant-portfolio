// Minimal check macro for the C++ pack: report the failing line and stop.
#pragma once
#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                             \
        }                                                                             \
    } while (0)

#define PASS(name) std::printf("PASS %s\n", name)
