# Equity options pricing engine: C++23 port of the Python eq_options package.
#
# Build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure
#   ./build/eqopt_bench
#
# Options:
#   EQOPT_WERROR   treat compiler warnings as errors (default OFF; CI sets ON)
#   EQOPT_TESTS    build the GoogleTest suite (default ON)
#   EQOPT_NATIVE   add -O3 -march=native to every target (default OFF; for
#                  local latency work only, never in CI: the binary will not
#                  run on older CPUs and FMA contraction can move last bits)
#   EQOPT_LTO      link-time optimisation through
#                  CMAKE_INTERPROCEDURAL_OPTIMIZATION (default OFF)
#   EQOPT_VEC_REPORT  print the compiler's vectorisation report for the
#                  batch kernels (-fopt-info-vec / -Rpass=loop-vectorize)
# Sanitizers and other instrumentation are passed through CMAKE_CXX_FLAGS
# and CMAKE_EXE_LINKER_FLAGS; this file never overwrites them.

cmake_minimum_required(VERSION 3.25)
project(equity_options_engine
        VERSION 1.1.0
        DESCRIPTION "Equity options pricing engine"
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

option(EQOPT_WERROR "Treat compiler warnings as errors" OFF)
option(EQOPT_TESTS "Build the test suite" ON)
option(EQOPT_NATIVE "Add -O3 -march=native (local benchmarking only)" OFF)
option(EQOPT_LTO "Enable link-time optimisation" OFF)
option(EQOPT_VEC_REPORT "Print the vectorisation report for batch kernels" OFF)

if(EQOPT_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT _eqopt_ipo OUTPUT _eqopt_ipo_msg)
  if(_eqopt_ipo)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "EQOPT_LTO requested but not supported: ${_eqopt_ipo_msg}")
  endif()
endif()

# Default to Release for single-config generators only; respect any choice.
get_property(_eqopt_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _eqopt_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Warning flags per compiler, attached to targets (not to CMAKE_CXX_FLAGS).
add_library(eqopt_warnings INTERFACE)
target_compile_options(eqopt_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
  $<$<AND:$<BOOL:${EQOPT_WERROR}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-Werror>
  $<$<AND:$<BOOL:${EQOPT_WERROR}>,$<CXX_COMPILER_ID:MSVC>>:/WX>)

# Tuning flags, attached to targets so CMAKE_CXX_FLAGS stays untouched.
add_library(eqopt_tuning INTERFACE)
target_compile_options(eqopt_tuning INTERFACE
  $<$<AND:$<BOOL:${EQOPT_NATIVE}>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3 -march=native>)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------- library ----
add_library(eqopt
  src/black_scholes.cpp
  src/greeks.cpp
  src/implied_vol.cpp
  src/binomial.cpp
  src/black76.cpp
  src/monte_carlo.cpp
  src/batch.cpp
  src/lr_tree.cpp
  src/bachelier.cpp
)
target_include_directories(eqopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(eqopt PUBLIC Threads::Threads eqopt_tuning PRIVATE eqopt_warnings)
# Shared numerical core (cpp/core, header-only target qp::core).
if(NOT TARGET qp::core)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core ${CMAKE_BINARY_DIR}/qp_core)
endif()
target_link_libraries(eqopt PUBLIC qp::core)
# The batch kernels: errno is never read, so -fno-math-errno lets the
# compiler inline sqrt and vectorise; it changes no returned value.
set_source_files_properties(src/batch.cpp PROPERTIES COMPILE_OPTIONS
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-math-errno>")
if(EQOPT_VEC_REPORT)
  set_property(SOURCE src/batch.cpp src/monte_carlo.cpp src/binomial.cpp APPEND PROPERTY COMPILE_OPTIONS
    "$<$<CXX_COMPILER_ID:GNU>:-fopt-info-vec-all>"
    "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Rpass=loop-vectorize;-Rpass-missed=loop-vectorize>")
endif()

# ------------------------------------------------------------- jsonl shim ----
# JSON-lines front end used by the cross-language differential test.
add_executable(eqopt_jsonl tools/jsonl_main.cpp)
target_link_libraries(eqopt_jsonl PRIVATE eqopt eqopt_warnings)

# -------------------------------------------------------------- benchmark ----
add_executable(eqopt_bench bench/bench_main.cpp)
target_link_libraries(eqopt_bench PRIVATE eqopt eqopt_warnings)
add_executable(eqopt_batch_bench tools/batch_bench.cpp)
target_link_libraries(eqopt_batch_bench PRIVATE eqopt eqopt_warnings)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _eqopt_bt)
set(_eqopt_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_eqopt_bt}}")
if(EQOPT_NATIVE)
  string(APPEND _eqopt_flags " -O3 -march=native")
endif()
if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  string(APPEND _eqopt_flags " LTO")
endif()
string(STRIP "${_eqopt_flags}" _eqopt_flags)
target_compile_definitions(eqopt_bench PRIVATE QP_BENCH_FLAGS="${_eqopt_flags}")

# ------------------------------------------------------------------ tests ----
# BUILD_TESTING=OFF (CTest convention) also turns the suite off.
if(EQOPT_TESTS AND (NOT DEFINED BUILD_TESTING OR BUILD_TESTING))
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

  add_executable(eqopt_tests
    tests/test_golden.cpp
    tests/test_black_scholes.cpp
    tests/test_greeks.cpp
    tests/test_implied_vol.cpp
    tests/test_binomial.cpp
    tests/test_black76.cpp
    tests/test_monte_carlo.cpp
    tests/test_numerical_safety.cpp
    tests/test_latency_api.cpp
    tests/test_phase5.cpp
  )
  target_include_directories(eqopt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(eqopt_tests PRIVATE eqopt eqopt_warnings GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(eqopt_tests)

  # Replaces global operator new to count allocations, so it is its own
  # binary. Sanitizer runtimes define operator new themselves (a static
  # runtime then clashes at link time), so sanitizer builds skip it.
  if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize")
    add_executable(eqopt_alloc_tests tests/test_no_alloc.cpp)
    target_link_libraries(eqopt_alloc_tests PRIVATE eqopt eqopt_warnings GTest::gtest GTest::gtest_main)
    gtest_discover_tests(eqopt_alloc_tests)
  endif()

  # The shim answers one good and one bad request.
  add_test(NAME eqopt_jsonl_smoke
    COMMAND ${CMAKE_COMMAND}
      -DSHIM=$<TARGET_FILE:eqopt_jsonl>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/jsonl_smoke_input.txt
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/jsonl_smoke.cmake)
endif()
