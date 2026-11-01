# FX options pricing engine: C++20 port of the Python fx_options package.
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
# Sanitizers and other instrumentation are passed through CMAKE_CXX_FLAGS
# and CMAKE_EXE_LINKER_FLAGS; this file never overwrites them.

cmake_minimum_required(VERSION 3.20)
project(fx_options_engine
        VERSION 1.1.0
        DESCRIPTION "FX options pricing engine"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FXOPT_WERROR "Treat compiler warnings as errors" OFF)
option(FXOPT_TESTS "Build the test suite" ON)

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
)
target_include_directories(fxopt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fxopt PRIVATE fxopt_warnings)

# ------------------------------------------------------------- jsonl shim ----
# JSON-lines front end used by the cross-language differential test.
add_executable(fxopt_jsonl tools/jsonl_main.cpp)
target_link_libraries(fxopt_jsonl PRIVATE fxopt fxopt_warnings)

# -------------------------------------------------------------- benchmark ----
add_executable(fxopt_bench bench/bench_main.cpp)
target_link_libraries(fxopt_bench PRIVATE fxopt fxopt_warnings)

# ------------------------------------------------------------------ tests ----
if(FXOPT_TESTS)
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
  )
  target_include_directories(fxopt_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_link_libraries(fxopt_tests PRIVATE fxopt fxopt_warnings GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(fxopt_tests)

  # The shim answers one good and one bad request.
  add_test(NAME fxopt_jsonl_smoke
    COMMAND ${CMAKE_COMMAND}
      -DSHIM=$<TARGET_FILE:fxopt_jsonl>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/jsonl_smoke_input.txt
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/jsonl_smoke.cmake)
endif()
