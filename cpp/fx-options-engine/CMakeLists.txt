cmake_minimum_required(VERSION 3.20)
project(fx_options_engine
        VERSION 1.0.0
        DESCRIPTION "High-performance FX options pricing engine (C++ twin of python/fx/01-options-pricing)"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Werror -O2)

# ---------------------------------------------------------------- library
add_library(fxopt STATIC
  src/common.cpp
  src/garman_kohlhagen.cpp
  src/forwards.cpp
  src/black76.cpp
  src/deltas.cpp
  src/greeks.cpp
  src/implied_vol.cpp
  src/binomial.cpp
  src/monte_carlo.cpp
)
target_include_directories(fxopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------------ tests
find_package(GTest REQUIRED)
enable_testing()

add_executable(fxopt_tests
  tests/test_garman_kohlhagen.cpp
  tests/test_golden.cpp
  tests/test_forwards_black76.cpp
  tests/test_symmetry.cpp
  tests/test_deltas.cpp
  tests/test_greeks.cpp
  tests/test_implied_vol.cpp
  tests/test_binomial.cpp
  tests/test_monte_carlo.cpp
  tests/test_edge_cases.cpp
)
target_link_libraries(fxopt_tests PRIVATE fxopt GTest::gtest_main)
target_include_directories(fxopt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

include(GoogleTest)
gtest_discover_tests(fxopt_tests)

# -------------------------------------------------------------- benchmark
add_executable(fxopt_bench bench/bench_main.cpp)
target_link_libraries(fxopt_bench PRIVATE fxopt)
