cmake_minimum_required(VERSION 3.25)
project(equity_var_engine CXX)

# C++23 floor. QP_CXX26=ON compiles with C++26 and turns on the features
# qp::core guards with feature-test macros (contracts, std::simd); the
# results are the same either way.
option(QP_CXX26 "Compile with C++26 and enable C++26 features where supported" OFF)
if(QP_CXX26)
  set(CMAKE_CXX_STANDARD 26)
else()
  set(CMAKE_CXX_STANDARD 23)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Respect the caller's build type; default to Release for single-config generators.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EQVAR_WERROR "Treat compiler warnings as errors" OFF)
option(EQVAR_BUILD_BENCH "Build the benchmark executable" ON)
# EQVAR_NATIVE adds -O3 -march=native to every target: local latency work
# only, never in CI (the binary will not run on older CPUs and FMA
# contraction can move last bits). EQVAR_LTO turns on link-time optimisation
# through CMAKE_INTERPROCEDURAL_OPTIMIZATION. EQVAR_VEC_REPORT prints the
# compiler's vectorisation report for the hot kernels. CMAKE_CXX_FLAGS is
# never modified.
option(EQVAR_NATIVE "Add -O3 -march=native (local benchmarking only)" OFF)
option(EQVAR_LTO "Enable link-time optimisation" OFF)
option(EQVAR_VEC_REPORT "Print the vectorisation report for the hot kernels" OFF)

if(EQVAR_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _eqvar_ipo OUTPUT _eqvar_ipo_msg)
  if(_eqvar_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "EQVAR_LTO requested but not supported: ${_eqvar_ipo_msg}")
  endif()
endif()

add_library(eqvar_tuning INTERFACE)
target_compile_options(eqvar_tuning INTERFACE
  $<$<AND:$<BOOL:${EQVAR_NATIVE}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3 -march=native>)

# Warnings: -Wall -Wextra -Wpedantic (GCC/Clang), /W4 /permissive- (MSVC).
set(EQVAR_WARNINGS
  $<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive->
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
  src/kernels.cpp
  src/option_reval.cpp
)
target_include_directories(eqvar PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_compile_options(eqvar PRIVATE ${EQVAR_WARNINGS})
target_link_libraries(eqvar PUBLIC eqvar_tuning)
# Shared numerical core (cpp/core, header-only target qp::core).
if(NOT TARGET qp::core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_BINARY_DIR}/qp_core)
endif()
target_link_libraries(eqvar PUBLIC qp::core)
if(EQVAR_VEC_REPORT)
  set_property(SOURCE src/kernels.cpp src/matrix.cpp APPEND PROPERTY COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:GNU>:-fopt-info-vec-all>"
    "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Rpass=loop-vectorize;-Rpass-missed=loop-vectorize>")
endif()

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
    test_oracle
    test_edge_cases
    test_fuzz
    test_conventions
    test_latency_api
    test_option_reval
    test_option_var
  )
  foreach(t IN LISTS EQVAR_TESTS)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE eqvar GTest::gtest GTest::gtest_main)
    target_compile_options(${t} PRIVATE ${EQVAR_WARNINGS}
      # EXPECT_THROW(f(x), ...) discards [[nodiscard]] results by design, and
      # the tests exercise the deprecated entry points on purpose.
      $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-unused-result;-Wno-deprecated-declarations>
      $<$<CXX_COMPILER_ID:MSVC>:/wd4834;/wd4996>)
    gtest_discover_tests(${t})
  endforeach()

  # Replaces global operator new to count allocations, so it is its own
  # binary. Sanitizer runtimes define operator new themselves (a static
  # runtime then clashes at link time), so sanitizer builds skip it.
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_executable(eqvar_alloc_tests tests/test_no_alloc.cpp)
    target_link_libraries(eqvar_alloc_tests PRIVATE eqvar GTest::gtest GTest::gtest_main)
    target_compile_options(eqvar_alloc_tests PRIVATE ${EQVAR_WARNINGS})
    gtest_discover_tests(eqvar_alloc_tests)
  endif()
endif()

# ---- benchmark --------------------------------------------------------------
if(EQVAR_BUILD_BENCH)
  add_executable(bench_eqvar bench/bench_main.cpp)
  target_link_libraries(bench_eqvar PRIVATE eqvar)
  target_compile_options(bench_eqvar PRIVATE ${EQVAR_WARNINGS})
  string(TOUPPER "${CMAKE_BUILD_TYPE}" _eqvar_bt)
  set(_eqvar_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_eqvar_bt}}")
  if(EQVAR_NATIVE)
    string(APPEND _eqvar_flags " -O3 -march=native")
  endif()
  if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    string(APPEND _eqvar_flags " LTO")
  endif()
  string(STRIP "${_eqvar_flags}" _eqvar_flags)
  target_compile_definitions(bench_eqvar PRIVATE QP_BENCH_FLAGS="${_eqvar_flags}")
endif()
