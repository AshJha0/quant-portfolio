# Equity Options Pricing Engine — C++20 twin of the Python eq_options package.
#
# Build:
#   cmake -S . -B build && cmake --build build -j
#   ctest --test-dir build --output-on-failure
#   ./build/eqopt_bench

cmake_minimum_required(VERSION 3.20)
project(equity_options_engine
        VERSION 1.0.0
        DESCRIPTION "High-performance equity options pricing engine"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Werror -O2)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- library ----
add_library(eqopt
  src/black_scholes.cpp
  src/greeks.cpp
  src/implied_vol.cpp
  src/binomial.cpp
  src/black76.cpp
  src/monte_carlo.cpp
)
target_include_directories(eqopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(eqopt PUBLIC Threads::Threads)

# ------------------------------------------------------------------ tests ----
enable_testing()
find_package(GTest REQUIRED)

add_executable(eqopt_tests
  tests/test_golden.cpp
  tests/test_black_scholes.cpp
  tests/test_greeks.cpp
  tests/test_implied_vol.cpp
  tests/test_binomial.cpp
  tests/test_black76.cpp
  tests/test_monte_carlo.cpp
)
target_include_directories(eqopt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(eqopt_tests PRIVATE eqopt GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(eqopt_tests)

# -------------------------------------------------------------- benchmark ----
add_executable(eqopt_bench bench/bench_main.cpp)
target_link_libraries(eqopt_bench PRIVATE eqopt)
