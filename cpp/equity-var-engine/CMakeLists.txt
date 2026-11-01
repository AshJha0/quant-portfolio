cmake_minimum_required(VERSION 3.20)
project(equity_var_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Respect the caller's build type; default to Release for single-config generators.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EQVAR_WERROR "Treat compiler warnings as errors" OFF)
option(EQVAR_BUILD_BENCH "Build the benchmark executable" ON)

# Warnings for GCC/Clang only; MSVC keeps its defaults.
set(EQVAR_WARNINGS
  $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall;-Wextra;-Wpedantic>
  $<$<AND:$<BOOL:${EQVAR_WERROR}>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-Werror>
  $<$<AND:$<BOOL:${EQVAR_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>
)

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
target_compile_options(eqvar PRIVATE ${EQVAR_WARNINGS})

# ---- tests ------------------------------------------------------------------
include(CTest)  # defines BUILD_TESTING (default ON)
if(BUILD_TESTING)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)

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
    test_fuzz
    test_conventions
  )
  foreach(t IN LISTS EQVAR_TESTS)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE eqvar GTest::gtest GTest::gtest_main)
    target_compile_options(${t} PRIVATE ${EQVAR_WARNINGS}
      # EXPECT_THROW(f(x), ...) discards [[nodiscard]] results by design, and
      # the tests exercise the deprecated entry points on purpose.
      $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-unused-result;-Wno-deprecated-declarations>)
    gtest_discover_tests(${t})
  endforeach()
endif()

# ---- benchmark --------------------------------------------------------------
if(EQVAR_BUILD_BENCH)
  add_executable(bench_eqvar bench/bench_main.cpp)
  target_link_libraries(bench_eqvar PRIVATE eqvar)
  target_compile_options(bench_eqvar PRIVATE ${EQVAR_WARNINGS})
endif()
