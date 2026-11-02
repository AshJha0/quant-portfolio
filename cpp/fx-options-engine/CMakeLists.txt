# FX options pricing engine: C++23 port of the Python fx_options package.
#
# Build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#   ./build/fxopt_bench
#
# Options:
#   FXOPT_WERROR   treat compiler warnings as errors (default OFF; CI sets ON)
#   FXOPT_TESTS    build the GoogleTest suite (default ON)
#   FXOPT_NATIVE   add -O3 -march=native to every target (default OFF; for
#                  local latency work only, never in CI: the binary will not
#                  run on older CPUs and FMA contraction can move last bits)
#   FXOPT_LTO      link-time optimisation through
#                  CMAKE_INTERPROCEDURAL_OPTIMIZATION (default OFF)
#   FXOPT_VEC_REPORT  print the compiler's vectorisation report for the
#                  batch, tree and Monte Carlo kernels
# Sanitizers and other instrumentation are passed through CMAKE_CXX_FLAGS
# and CMAKE_EXE_LINKER_FLAGS; this file never overwrites them.

cmake_minimum_required(VERSION 3.25)
project(fx_options_engine
        VERSION 1.1.0
        DESCRIPTION "FX options pricing engine"
        LANGUAGES CXX)

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

option(FXOPT_WERROR "Treat compiler warnings as errors" OFF)
option(FXOPT_TESTS "Build the test suite" ON)
option(FXOPT_NATIVE "Add -O3 -march=native (local benchmarking only)" OFF)
option(FXOPT_LTO "Enable link-time optimisation" OFF)
option(FXOPT_VEC_REPORT "Print the vectorisation report for batch kernels" OFF)

if(FXOPT_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _fxopt_ipo OUTPUT _fxopt_ipo_msg)
  if(_fxopt_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "FXOPT_LTO requested but not supported: ${_fxopt_ipo_msg}")
  endif()
endif()

# Default to Release for single-config generators only; respect any choice.
get_property(_fxopt_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _fxopt_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Warning flags per compiler, attached to targets (not to CMAKE_CXX_FLAGS).
add_library(fxopt_warnings INTERFACE)
target_compile_options(fxopt_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
  $<$<AND:$<BOOL:${FXOPT_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>
  $<$<AND:$<BOOL:${FXOPT_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>)

# Tuning flags, attached to targets so CMAKE_CXX_FLAGS stays untouched.
add_library(fxopt_tuning INTERFACE)
target_compile_options(fxopt_tuning INTERFACE
  $<$<AND:$<BOOL:${FXOPT_NATIVE}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3 -march=native>)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- library ----
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
  src/batch.cpp
)
target_include_directories(fxopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fxopt PUBLIC Threads::Threads fxopt_tuning PRIVATE fxopt_warnings)
# Shared numerical core (cpp/core, header-only target qp::core).
if(NOT TARGET qp::core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_BINARY_DIR}/qp_core)
endif()
target_link_libraries(fxopt PUBLIC qp::core)
# The batch kernels: errno is never read, so -fno-math-errno lets the
# compiler inline sqrt and vectorise; it changes no returned value.
set_source_files_properties(src/batch.cpp PROPERTIES COMPILE_OPTIONS
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-math-errno>")
if(FXOPT_VEC_REPORT)
  set_property(SOURCE src/batch.cpp src/monte_carlo.cpp src/binomial.cpp APPEND PROPERTY COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:GNU>:-fopt-info-vec-all>"
    "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Rpass=loop-vectorize;-Rpass-missed=loop-vectorize>")
endif()

# ------------------------------------------------------------- jsonl shim ----
# JSON-lines front end used by the cross-language differential test.
add_executable(fxopt_jsonl tools/jsonl_main.cpp)
target_link_libraries(fxopt_jsonl PRIVATE fxopt fxopt_warnings)

# -------------------------------------------------------------- benchmark ----
add_executable(fxopt_bench bench/bench_main.cpp)
target_link_libraries(fxopt_bench PRIVATE fxopt fxopt_warnings)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _fxopt_bt)
set(_fxopt_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_fxopt_bt}}")
if(FXOPT_NATIVE)
  string(APPEND _fxopt_flags " -O3 -march=native")
endif()
if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  string(APPEND _fxopt_flags " LTO")
endif()
string(STRIP "${_fxopt_flags}" _fxopt_flags)
target_compile_definitions(fxopt_bench PRIVATE QP_BENCH_FLAGS="${_fxopt_flags}")

# ------------------------------------------------------------------ tests ----
# BUILD_TESTING=OFF (CTest convention) also turns the suite off.
if(FXOPT_TESTS AND (NOT DEFINED BUILD_TESTING OR BUILD_TESTING))
  enable_testing()
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    find_package(GTest QUIET)
  endif()
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

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
    tests/test_numerical_safety.cpp
    tests/test_latency_api.cpp
  )
  target_include_directories(fxopt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(fxopt_tests PRIVATE fxopt fxopt_warnings GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(fxopt_tests)

  # Replaces global operator new to count allocations, so it is its own
  # binary. Sanitizer runtimes define operator new themselves (a static
  # runtime then clashes at link time), so sanitizer builds skip it.
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_executable(fxopt_alloc_tests tests/test_no_alloc.cpp)
    target_link_libraries(fxopt_alloc_tests PRIVATE fxopt fxopt_warnings GTest::gtest GTest::gtest_main)
    gtest_discover_tests(fxopt_alloc_tests)
  endif()

  # The shim answers one good and one bad request.
  add_test(NAME fxopt_jsonl_smoke
    COMMAND ${CMAKE_COMMAND}
      -DSHIM=$<TARGET_FILE:fxopt_jsonl>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/jsonl_smoke_input.txt
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/jsonl_smoke.cmake)
endif()
