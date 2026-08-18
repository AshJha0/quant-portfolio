cmake_minimum_required(VERSION 3.20)
project(equity_var_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Werror -O2)

# ---- library ----------------------------------------------------------------
add_library(eqvar STATIC
  src/matrix.cpp
  src/stats.cpp
  src/returns.cpp
  src/historical.cpp
  src/parametric.cpp
  src/expected_shortfall.cpp
  src/monte_carlo.cpp
  src/backtest.cpp
)
target_include_directories(eqvar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- tests ------------------------------------------------------------------
find_package(GTest REQUIRED)
include(GoogleTest)
enable_testing()

set(EQVAR_TESTS
  test_stats
  test_matrix
  test_historical
  test_parametric
  test_expected_shortfall
  test_monte_carlo
  test_backtest
  test_cross_language
  test_edge_cases
)

foreach(t IN LISTS EQVAR_TESTS)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE eqvar GTest::gtest GTest::gtest_main)
  # EXPECT_THROW(f(x), ...) legitimately discards [[nodiscard]] results when
  # asserting that f throws; the library itself still builds with full -Werror.
  target_compile_options(${t} PRIVATE -Wno-unused-result)
  gtest_discover_tests(${t})
endforeach()

# ---- benchmark --------------------------------------------------------------
add_executable(bench_eqvar bench/bench_main.cpp)
target_link_libraries(bench_eqvar PRIVATE eqvar)
