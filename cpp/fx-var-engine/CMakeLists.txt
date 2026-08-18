cmake_minimum_required(VERSION 3.20)
project(fx_var_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Werror -O2)

# ------------------------------------------------------------------ library
add_library(fxvar
  src/matrix.cpp
  src/stats.cpp
  src/returns.cpp
  src/book.cpp
  src/expected_shortfall.cpp
  src/historical.cpp
  src/parametric.cpp
  src/monte_carlo.cpp
  src/backtest.cpp
  src/stress.cpp
)
target_include_directories(fxvar PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# -------------------------------------------------------------------- tests
find_package(GTest REQUIRED)
include(GoogleTest)
enable_testing()

set(FXVAR_TESTS
  test_matrix_stats
  test_book
  test_expected_shortfall
  test_historical
  test_parametric
  test_monte_carlo
  test_backtest
  test_stress
  test_golden_python
)

foreach(t ${FXVAR_TESTS})
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE fxvar GTest::gtest_main)
  gtest_discover_tests(${t})
endforeach()

# -------------------------------------------------------------------- bench
add_executable(fxvar_bench bench/bench_main.cpp)
target_link_libraries(fxvar_bench PRIVATE fxvar)
